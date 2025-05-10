# SDN Controller

## Features

- Network Topology Management: Add and remove switches and links dynamically
- Traffic Engineering Policies:
  - Load-balancing across multiple paths
  - Traffic prioritization based on type
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
python q4_sdn_controller.py
```
