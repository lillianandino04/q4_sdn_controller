# SDN Controller

## Features

- Network Topology Management: Add and remove switches and links dynamically
- Traffic Engineering Policies:
  - Load-balancing across multiple paths
  - Traffic prioritization based on type (Critical, Business, Normal, Background)
  - Automatic computation of backup paths for resilience
- Interactive Visualization: View the network topology with link utilization in real-time
- Failure Handling: Simulate link failures and recoveries with automatic flow rerouting
- Command-Line Interface: Intuitive commands for network management and monitoring

## Dependencies

- Python 3.7 or higher
- NetworkX
- Matplotlib
  USE:
- networkx>=2.5
- matplotlib>=3.3.0
- numpy>=1.19.0
- cmd2>=1.0.0

## Installation

1. Clone this repository
2. Install dependencies


Run the SDN controller with:

```bash
python sdn_controller.py
```

#### Network Management

- `add_switch <switch_id>` - Add a new switch
- `remove_switch <switch_id>` - Remove a switch
- `add_link <source> <target> [capacity] [delay] [cost] [bidirectional]` - Add a link between switches
- `remove_link <source> <target> [bidirectional]` - Remove a link

#### Flow Management

- `add_flow <source> <destination> <traffic_type> [bandwidth]` - Add a new flow
  - Traffic types: CRITICAL, BUSINESS, NORMAL, BACKGROUND
- `remove_flow <flow_id>` - Remove a flow

#### Network Visualization and Information

- `show_topology [save_path]` - Visualize the network topology (optionally save to file)
- `show_network_status` - Show current network statistics
- `show_flows` - List all active flows
- `show_flow <flow_id>` - Show details of a specific flow
- `show_switch <switch_id>` - Show details of a specific switch

#### Failure Simulation

- `fail_link <source> <target>` - Simulate a link failure
- `recover_link <source> <target>` - Simulate a link recovery

#### Testing and Configuration

- `create_test_network` - Create a predefined test topology
- `create_test_flows` - Create some test flows on the network
- `save_topology <filename>` - Save the current topology to a JSON file
- `load_topology <filename>` - Load a topology from a JSON file
