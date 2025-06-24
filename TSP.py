import requests
import pandas as pd  # Import pandas for Excel operations
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import json
import urllib.parse

# --- 1. Initial Setup ---
ORS_API_KEY = "5b3ce3597851110001cf6248d2f4c6b4ac0c464c82c0d9a9247eec54"  # Replace with your Openrouteservice API Key
ORS_BASE_URL = "https://api.openrouteservice.org"
EXCEL_FILE_PATH = "TSP_Data.xlsx"  # Path to your Excel file


# --- 2. Load Data from Excel ---
def load_data_from_excel(file_path):
    """
    Loads vehicle start/end points, work points, vehicle capacities, and work point loads
    from an Excel file.
    Expects two sheets: 'MachineLocations' and 'WorkPoints'.
    """
    try:
        df_machines = pd.read_excel(file_path, sheet_name='MachineLocations')
        df_work_points = pd.read_excel(file_path, sheet_name='WorkPoints')
    except FileNotFoundError:
        print(f"Error: Excel file not found at {file_path}")
        return None, None, None, None, None, None, None
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None, None, None, None, None, None, None

    points = {}  # Dictionary to store all point names and their (lat, lon) coordinates
    starts = []  # List of start point names for each vehicle
    ends = []  # List of end point names for each vehicle
    num_vehicles = len(df_machines)
    vehicle_capacities = []  # List of capacities for each vehicle

    # Process machine locations and capacities
    for index, row in df_machines.iterrows():
        start_name = row['StartPointName']
        end_name = row['EndPointName']

        points[start_name] = (row['StartLat'], row['StartLon'])
        points[end_name] = (row['EndLat'], row['EndLon'])
        starts.append(start_name)
        ends.append(end_name)
        vehicle_capacities.append(row['Capacity'])  # Read vehicle capacity

    # Process work points and their loads
    work_points_names = []
    # This will store the 'demand' for each location, including zero for depots
    demands = [0] * len(points)

    # Temporarily map names to indices to assign demands correctly
    # Create a temporary list of all point names in the order they'll appear in the matrix
    all_temp_names = list(points.keys())

    for index, row in df_work_points.iterrows():
        point_name = row['PointName']
        point_lat = row['Lat']
        point_lon = row['Lon']
        point_load = row['Load']

        # Add to the main points dictionary if it's not already there (e.g., if a work point is also a start/end)
        if point_name not in points:
            points[point_name] = (point_lat, point_lon)

        work_points_names.append(point_name)

        # Add demand to the demands list at the correct index
        # We need to ensure that the demand list corresponds to the time_matrix order later
        # For now, let's just make sure all points are in `points` and we'll build demand list relative to `locations_names_from_ors_api` later

    print("Data loaded successfully from Excel.")
    return points, starts, ends, work_points_names, num_vehicles, vehicle_capacities


# --- 3. Function to get travel time matrix from Openrouteservice Distance Matrix API ---
def get_time_matrix_ors(locations_dict, api_key):
    """
    Uses Openrouteservice Distance Matrix API to get travel times between all pairs of points.
    Returns a 2D list (matrix) of durations in seconds.
    """
    locations_list = list(locations_dict.values())
    locations_names = list(locations_dict.keys())

    # ORS requires coordinates in [longitude, latitude] format.
    ors_coords = [[lon, lat] for lat, lon in locations_list]

    headers = {
        'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png',
        'Content-Type': 'application/json; charset=utf-8',
        'Authorization': api_key
    }

    # URL for ORS Matrix API for car profiles
    url = f"{ORS_BASE_URL}/v2/matrix/driving-car"

    # POST request payload parameters
    payload = {
        "locations": ors_coords,
        "metrics": ["duration"],  # Request only duration
        "sources": list(range(len(ors_coords))),  # All points as origins
        "destinations": list(range(len(ors_coords))),  # All points as destinations
        "units": "s"  # Time unit in seconds
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        result = response.json()

        if 'durations' in result:
            time_matrix = result['durations']  # Durations are returned in seconds
            print("Travel time matrix successfully retrieved from ORS.")
            return time_matrix, locations_names
        else:
            print(f"ORS response does not contain 'durations'. Full response: {json.dumps(result, indent=2)}")
            return None, None

    except requests.exceptions.ConnectionError:
        print(f"Error connecting to ORS server. Ensure internet connection is stable.")
        return None, None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error from ORS server: {e}")
        print(f"Server response: {response.text}")
        if response.status_code == 429:
            print(
                "You likely hit Openrouteservice free plan limits (Too Many Requests). Please wait or upgrade your plan.")
        return None, None
    except Exception as e:
        print(f"Error getting time matrix from ORS API: {e}")
        return None, None


# --- 4. Function to prepare data for OR-Tools ---
def create_data_model(time_matrix, starts_indices, ends_indices, num_vehicles, all_locations_names,
                      vehicle_capacities, work_points_data_from_excel):
    """
    Prepares the data model for OR-Tools, including time matrix, demands, and capacities.
    """
    data = {}
    data['time_matrix'] = time_matrix
    data['num_vehicles'] = num_vehicles
    data['depots'] = [starts_indices[i] for i in range(num_vehicles)]  # Start nodes for each vehicle
    data['ends'] = [ends_indices[i] for i in range(num_vehicles)]  # End nodes for each vehicle
    data['locations_names'] = all_locations_names  # All location names (for mapping indices back to names)

    # Prepare demands list based on the order of locations_names_from_ors_api
    demands = [0] * len(all_locations_names)
    for index, row in work_points_data_from_excel.iterrows():
        point_name = row['PointName']
        point_load = row['Load']
        if point_name in all_locations_names:  # Ensure the work point is in the matrix locations
            demand_index = all_locations_names.index(point_name)
            demands[demand_index] = point_load
        else:
            print(f"Warning: Work point '{point_name}' not found in locations from ORS. Demand ignored.")

    data['demands'] = demands
    data['vehicle_capacities'] = vehicle_capacities

    return data


# --- 5. Function to solve the VRP with OR-Tools ---
def solve_vrp(data, work_points_names_for_vrp_disjunction):
    """
    Solves the Vehicle Routing Problem (VRP) using OR-Tools,
    including time and capacity constraints.
    """
    manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                           data['num_vehicles'],
                                           data['depots'],
                                           data['ends'])

    routing = pywrapcp.RoutingModel(manager)

    # Define the transit callback for travel time
    def time_callback(from_index, to_index):
        """Returns the travel time between two nodes based on the time matrix."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add a Dimension for TIME to track total route duration (optional but good practice)
    # Allows setting max travel time per vehicle if needed
    # routing.AddDimension(
    #     transit_callback_index,
    #     0,      # slack_max (maximum waiting time at a node)
    #     28800,  # vehicle_capacity (max total time in seconds, e.g., 8 hours)
    #     True,   # fix_start_cumul_to_zero (cumulated time starts at 0 for each vehicle)
    #     'Time')
    # time_dimension = routing.GetDimensionOrDie('Time')

    # Define the demand callback for load volume
    def demand_callback(from_index):
        """Returns the demand (load) of the node."""
        node = manager.IndexToNode(from_index)
        return data['demands'][node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    # Add a Capacity Dimension to enforce vehicle load limits
    routing.AddDimensionWithVehicleCapacities(
        demand_callback_index,
        0,  # null_capacity_slack: how much capacity can be unused at a node (0 means no slack needed)
        data['vehicle_capacities'],  # Capacities for each vehicle
        True,  # fix_start_cumul_to_zero: current load starts at 0 for each vehicle
        'Capacity')  # Name of the dimension
    capacity_dimension = routing.GetDimensionOrDie('Capacity')

    # Add work points as mandatory visits (disjunctions)
    # This ensures that each work point is visited by exactly one vehicle.
    for work_point_name in work_points_names_for_vrp_disjunction:
        # Get the numerical index of the work point from the ORS API's order
        work_point_idx = data['locations_names'].index(work_point_name)
        # Add a disjunction: this point must be visited by one of the routes.
        # A cost of 0 means not visiting this point is not acceptable.
        routing.AddDisjunction([manager.NodeToIndex(work_point_idx)], 0)

        # Set search parameters for OR-Tools
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)  # Initial solution strategy

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)  # Metaheuristic for solution improvement

    # Time limit for search (you can increase this for better solutions, but it will take longer)
    search_parameters.time_limit.FromSeconds(10)

    # Solve the model
    solution = routing.SolveWithParameters(search_parameters)

    return manager, routing, solution


# --- 6. Function to print results and generate Openrouteservice Map links ---
def print_solution(data, manager, routing, solution, all_original_points_data, all_locations_names):
    """
    Prints the results and generates Openrouteservice Map links for each vehicle's route.
    Also prints the load carried by each vehicle.
    """
    if not solution:
        print("No solution found by OR-Tools.")
        return

    total_time_all_vehicles = 0
    print(f"\n--- Routing Solution ---")
    print(f"Total number of points considered: {len(all_locations_names)}")
    print(f"Number of vehicles: {data['num_vehicles']}")
    print(f"Vehicle start points (depots): {[all_locations_names[d] for d in data['depots']]}")
    print(f"Vehicle end points: {[all_locations_names[e] for e in data['ends']]}")

    capacity_dimension = routing.GetDimensionOrDie('Capacity')

    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route_nodes_indices = []  # Numerical indices of points in the route
        route_names = []  # Names of points in the route
        current_load = 0

        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_nodes_indices.append(node_index)
            route_names.append(all_locations_names[node_index])
            # Add current node's demand to the load
            current_load += data['demands'][node_index]
            index = solution.Value(routing.NextVar(index))

        # Add the end point (last node in the route)
        node_index = manager.IndexToNode(index)
        route_nodes_indices.append(node_index)
        route_names.append(all_locations_names[node_index])
        # The load at the end point should ideally be 0 if all deliveries are made,
        # but for calculation purposes, we add its 'demand' which is 0 for end points.
        current_load += data['demands'][node_index]  # This will be 0 for end points

        # Calculate total travel time for the current route based on the time matrix
        current_route_total_time = 0
        for i in range(len(route_nodes_indices) - 1):
            from_node = route_nodes_indices[i]
            to_node = route_nodes_indices[i + 1]
            current_route_total_time += data['time_matrix'][from_node][to_node]

        print(f'\n--- Schedule for Vehicle {vehicle_id + 1} ---')
        print(f'Route: {" -> ".join(route_names)}')
        print(f'Total route time: {current_route_total_time / 60:.2f} minutes')  # Convert to minutes
        print(f'Vehicle Capacity: {data["vehicle_capacities"][vehicle_id]}')

        # OR-Tools Dimension Cumulative Value at the end of the route.
        # This will show the maximum load carried on the route segment.
        route_load_at_end = solution.CumulVarValue(capacity_dimension.CumulVar(routing.End(vehicle_id)))
        print(f'Total load assigned (at end of route, if returned to 0): {route_load_at_end}')

        # A more intuitive way to show total load delivered by the vehicle
        total_delivered_load = sum(data['demands'][manager.IndexToNode(idx)] for idx in route_nodes_indices if
                                   idx not in [data['depots'][vehicle_id], data['ends'][vehicle_id]])
        print(f'Total load delivered by this vehicle: {total_delivered_load}')

        total_time_all_vehicles += current_route_total_time

        # Generate Openrouteservice Directions link to visualize the route on the map
        route_coords_ors_format = ";".join(
            [f"{all_original_points_data[name][1]},{all_original_points_data[name][0]}" for name in route_names])

        center_lat, center_lon = all_original_points_data[route_names[0]]
        ors_map_url = (f"https://maps.openrouteservice.org/directions?n1={center_lat:.4f}&n2={center_lon:.4f}&b=1"
                       f"&c=0&a={route_coords_ors_format}&r=1&k1=false&k2=false&k3=false&k4=false&k5=false&h=0")

        encoded_ors_map_url = urllib.parse.quote(ors_map_url, safe=':/,&=?')

        print(f'Link to Openrouteservice Maps: {encoded_ors_map_url}')

    print(f'\nTotal operation time for all vehicles combined: {total_time_all_vehicles / 60:.2f} minutes')


# --- 7. Main execution of the program ---
if __name__ == '__main__':
    # Step 1: Load all points data, vehicle starts/ends, capacities, and work points with loads from Excel
    all_points_data, vehicle_starts_names, vehicle_ends_names, work_points_to_visit, num_of_vehicles, vehicle_capacities = load_data_from_excel(
        EXCEL_FILE_PATH)

    if all_points_data is None:
        print("Exiting due to data loading error.")
    else:
        # Step 2: Get time matrix from Openrouteservice for ALL points (including start/end/work)
        time_matrix, locations_names_from_ors_api = get_time_matrix_ors(all_points_data, ORS_API_KEY)

        if time_matrix and locations_names_from_ors_api:
            # We also need the actual DataFrame of work points to map loads
            df_work_points_for_demands = pd.read_excel(EXCEL_FILE_PATH, sheet_name='WorkPoints')

            # Map location names back to their numerical indices for OR-Tools
            start_indices = [locations_names_from_ors_api.index(s) for s in vehicle_starts_names]
            end_indices = [locations_names_from_ors_api.index(e) for e in vehicle_ends_names]

            data_model = create_data_model(time_matrix, start_indices, end_indices, num_of_vehicles,
                                           locations_names_from_ors_api, vehicle_capacities, df_work_points_for_demands)

            manager, routing_model, solution_found = solve_vrp(data_model, work_points_to_visit)
            print_solution(data_model, manager, routing_model, solution_found, all_points_data,
                           locations_names_from_ors_api)
        else:
            print(
                "Failed to retrieve time matrix from Openrouteservice. Please check your API key and network connection.")