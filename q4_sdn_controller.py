#!/usr/bin/env python3
"""
Student ID: 893353640
Cryptographic Watermark: 0c7ae0ad44a5d45a07ca84902b80f20a16a6f5a4e4c6fd6dab7a5a73fecaadee
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
import hashlib
import time
import json
from collections import defaultdict, Counter
from enum import Enum
import cmd
import threading
import queue

#WATERMARK = "0c7ae0ad44a5d45a07ca84902b80f20a16a6f5a4e4c6fd6dab7a5a73fecaadee"

class TrafficType(Enum):
    """Traffic types with priority levels"""
    CRITICAL = 3
    BUSINESS = 2
    NORMAL = 1
    BACKGROUND = 0

class Flow:
    """
    Represents a network flow between source and destination
    with specific characteristics
    """
    def __init__(self, flow_id, source, destination, traffic_type=TrafficType.NORMAL, 
                 bandwidth=1, path=None):
        self.flow_id = flow_id
        self.source = source
        self.destination = destination
        self.traffic_type = traffic_type
        self.bandwidth = bandwidth
        self.path = path or []
        self.is_active = True
        self.creation_time = time.time()
    
    def __str__(self):
        return (f"Flow {self.flow_id}: {self.source} → {self.destination} "
                f"[{self.traffic_type.name}, BW={self.bandwidth}]")

class FlowTableEntry:
    """Represents an entry in a switch's flow table"""
    def __init__(self, match_fields, actions, priority=1, timeout=None):
        self.match_fields = match_fields
        self.actions = actions
        self.priority = priority
        self.timeout = timeout
        self.creation_time = time.time()
    
    def __str__(self):
        fields_str = ', '.join(f"{k}={v}" for k, v in self.match_fields.items())
        actions_str = ', '.join(str(a) for a in self.actions)
        return f"Match: [{fields_str}] → Actions: [{actions_str}] (Priority: {self.priority})"

class Switch:
    """Represents an OpenFlow switch in the network"""
    def __init__(self, switch_id):
        self.switch_id = switch_id
        self.flow_table = []
        self.connected_links = set()
        self.port_map = {}
    
    def add_flow_entry(self, entry):
        """Add a flow entry to the switch's flow table"""
        self.flow_table.append(entry)
        #sort flow table by priority (higher numbers first)
        self.flow_table.sort(key=lambda entry: entry.priority, reverse=True)
    
    def remove_flow_entries(self, match_dict=None):
        """Remove flow entries matching the given criteria"""
        if not match_dict:
            self.flow_table = []
            return
            
        new_table = []
        for entry in self.flow_table:
            #check if all match_dict items are in the entry's match_fields
            if not all(entry.match_fields.get(k) == v for k, v in match_dict.items()):
                new_table.append(entry)
        self.flow_table = new_table
    
    def __str__(self):
        return f"Switch {self.switch_id} - {len(self.flow_table)} flow entries"

class Link:
    """Represents a network link between two switches"""
    def __init__(self, source, target, capacity=10, delay=1, cost=1):
        self.source = source
        self.target = target
        self.capacity = capacity
        self.delay = delay
        self.cost = cost
        self.utilization = 0
        self.flows = set()
    
    def add_flow(self, flow):
        """Add a flow to this link and update utilization"""
        self.flows.add(flow.flow_id)
        self.utilization += flow.bandwidth
    
    def remove_flow(self, flow):
        """Remove a flow from this link and update utilization"""
        if flow.flow_id in self.flows:
            self.flows.remove(flow.flow_id)
            self.utilization = max(0, self.utilization - flow.bandwidth)
    
    def get_utilization_percentage(self):
        """Get link utilization as a percentage"""
        return (self.utilization / self.capacity) * 100 if self.capacity > 0 else 0
    
    def __str__(self):
        return (f"Link {self.source} → {self.target} "
                f"[{self.utilization}/{self.capacity} Gbps, {self.delay}ms]")

class SDNController:
    """
    The main SDN Controller class that manages the network topology,
    computes routes, and configures switches
    """
    def __init__(self):
        #initialize network topology as a directed graph
        self.topology = nx.DiGraph()
        
        #maps to store network elements
        self.switches = {}  # switch_id -> Switch
        self.links = {}  # (source_id, target_id) -> Link
        self.flows = {}  # flow_id -> Flow
        
        #traffic statistics
        self.stats = {
            'total_flows': 0,
            'active_flows': 0,
            'dropped_packets': 0,
            'routed_packets': 0
        }
        
        #flow counter for generating IDs
        self.flow_counter = 0
        
        #queue for events (link failures, new flows, etc.)
        self.event_queue = queue.Queue()
        
        #start event processing thread
        self.stop_event = threading.Event()
        self.event_thread = threading.Thread(target=self._process_events)
        self.event_thread.daemon = True
        self.event_thread.start()
    
    def _process_events(self):
        """Background thread for processing network events"""
        while not self.stop_event.is_set():
            try:
                # Get event from queue with timeout
                event = self.event_queue.get(timeout=1)
                event_type = event.get('type')
                
                if event_type == 'link_failure':
                    self._handle_link_failure(event.get('link'))
                elif event_type == 'link_recovery':
                    self._handle_link_recovery(event.get('link'))
                elif event_type == 'new_flow':
                    self._handle_new_flow(event.get('flow'))
                
                self.event_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error processing event: {e}")
    
    def _handle_link_failure(self, link_id):
        """Handle a link failure event"""
        if link_id in self.links:
            source, target = link_id
            print(f"Handling link failure: {source} → {target}")
            
            #get flows affected by this link
            affected_flows = []
            for flow in self.flows.values():
                if flow.is_active and len(flow.path) > 1:
                    # Check if link is in the flow's path
                    for i in range(len(flow.path) - 1):
                        if (flow.path[i], flow.path[i+1]) == link_id:
                            affected_flows.append(flow)
                            break
            
            #remove the link from topology
            if self.topology.has_edge(source, target):
                self.topology.remove_edge(source, target)
            
            #update switches
            if source in self.switches:
                self.switches[source].connected_links.discard(link_id)
            if target in self.switches:
                self.switches[target].connected_links.discard((target, source))
            
            #reroute affected flows
            for flow in affected_flows:
                self.remove_flow(flow.flow_id)
                new_path = self.compute_path(flow.source, flow.destination, flow.traffic_type)
                if new_path:
                    flow.path = new_path
                    self.setup_flow_path(flow)
                else:
                    flow.is_active = False
                    self.stats['active_flows'] -= 1
                    print(f"No alternative path for flow {flow.flow_id}, flow removed")
    
    def _handle_link_recovery(self, link_id):
        """Handle a link recovery event"""
        source, target = link_id
        print(f"Handling link recovery: {source} → {target}")
        
        #add link back to topology
        if source in self.switches and target in self.switches:
            link = self.links.get(link_id)
            if link:
                self.topology.add_edge(source, target, 
                                    weight=link.cost,
                                    capacity=link.capacity,
                                    delay=link.delay)
                
                #update switches
                self.switches[source].connected_links.add(link_id)
                self.switches[target].connected_links.add((target, source))
            
                inactive_flows = [f for f in self.flows.values() 
                                 if not f.is_active and 
                                 f.source == source and f.destination == target]
                
                for flow in inactive_flows:
                    path = self.compute_path(flow.source, flow.destination, flow.traffic_type)
                    if path:
                        flow.path = path
                        flow.is_active = True
                        self.stats['active_flows'] += 1
                        self.setup_flow_path(flow)
                        print(f"Flow {flow.flow_id} has been reactivated")
    
    def _handle_new_flow(self, flow):
        """Handle a new flow event"""
        if flow and flow.flow_id not in self.flows:
            self.add_flow(flow)
    
    def add_switch(self, switch_id):
        """Add a new switch to the network"""
        if switch_id not in self.switches:
            switch = Switch(switch_id)
            self.switches[switch_id] = switch
            self.topology.add_node(switch_id)
            print(f"Added switch: {switch_id}")
            return switch
        return self.switches[switch_id]
    
    def remove_switch(self, switch_id):
        """Remove a switch from the network"""
        if switch_id in self.switches:
            #remove all connected links
            links_to_remove = []
            for link_id in self.links:
                src, dst = link_id
                if src == switch_id or dst == switch_id:
                    links_to_remove.append(link_id)
            
            for link_id in links_to_remove:
                self.remove_link(*link_id)
            
            #remove switch
            del self.switches[switch_id]
            if self.topology.has_node(switch_id):
                self.topology.remove_node(switch_id)
            
            print(f"Removed switch: {switch_id}")
            return True
        return False
    
    def add_link(self, source, target, capacity=10, delay=1, cost=1, bidirectional=True):
        """Add a new link between switches"""
        #ensure both switches exist
        src_switch = self.add_switch(source) if source not in self.switches else self.switches[source]
        dst_switch = self.add_switch(target) if target not in self.switches else self.switches[target]
        
        #create forward link
        link = Link(source, target, capacity, delay, cost)
        self.links[(source, target)] = link
        self.topology.add_edge(source, target, weight=cost, capacity=capacity, delay=delay)
        src_switch.connected_links.add((source, target))
        
        #set up port mappings
        port_num = len(src_switch.port_map) + 1
        src_switch.port_map[target] = port_num
        
        if bidirectional:
            rev_link = Link(target, source, capacity, delay, cost)
            self.links[(target, source)] = rev_link
            self.topology.add_edge(target, source, weight=cost, capacity=capacity, delay=delay)
            dst_switch.connected_links.add((target, source))
            
            #set up port mappings for reverse direction
            port_num = len(dst_switch.port_map) + 1
            dst_switch.port_map[source] = port_num
        
        print(f"Added link: {source} ↔ {target}" if bidirectional else f"Added link: {source} → {target}")
        return link
    
    def remove_link(self, source, target, bidirectional=True):
        """Remove a link from the network"""
        link_id = (source, target)
        rev_link_id = (target, source)
        
        if link_id in self.links:
            affected_flows = []
            for flow in self.flows.values():
                if flow.is_active and len(flow.path) > 1:
                    for i in range(len(flow.path) - 1):
                        if (flow.path[i], flow.path[i+1]) == link_id:
                            affected_flows.append(flow)
                            break
            
            #update links and topology
            if source in self.switches:
                self.switches[source].connected_links.discard(link_id)
            if self.topology.has_edge(*link_id):
                self.topology.remove_edge(*link_id)
            del self.links[link_id]
            
            #remove reverse link if bidirectional
            if bidirectional and rev_link_id in self.links:
                if target in self.switches:
                    self.switches[target].connected_links.discard(rev_link_id)
                if self.topology.has_edge(*rev_link_id):
                    self.topology.remove_edge(*rev_link_id)
                del self.links[rev_link_id]
            
            #reroute affected flows
            for flow in affected_flows:
                self.remove_flow(flow.flow_id)
                new_path = self.compute_path(flow.source, flow.destination, flow.traffic_type)
                if new_path:
                    flow.path = new_path
                    self.setup_flow_path(flow)
                else:
                    flow.is_active = False
                    self.stats['active_flows'] -= 1
            
            print(f"Removed link: {source} ↔ {target}" if bidirectional else f"Removed link: {source} → {target}")
            return True
        return False
    
    def compute_path(self, source, destination, traffic_type=TrafficType.NORMAL):
        if source not in self.topology.nodes or destination not in self.topology.nodes:
            return []
        
        if source == destination:
            return [source]
        
        try:
            if traffic_type == TrafficType.CRITICAL:
                path = self._find_least_utilized_path(source, destination)
                if not path:
                    path = nx.shortest_path(self.topology, source, destination, weight='weight')
            
            elif traffic_type == TrafficType.BUSINESS:
                path = nx.shortest_path(self.topology, source, destination, weight='delay')
            
            elif traffic_type == TrafficType.NORMAL:
                path = self._find_load_balanced_path(source, destination)
            
            else:
                path = nx.shortest_path(self.topology, source, destination, weight='weight')
                
            return path
        
        except nx.NetworkXNoPath:
            return []
        except Exception as e:
            print(f"Error computing path: {e}")
            return []
    
    def _find_least_utilized_path(self, source, destination):
        """Find path with least utilized links"""
        try:
            G = self.topology.copy()
            
            #assign weights based on utilization
            for u, v, data in G.edges(data=True):
                link = self.links.get((u, v))
                if link:
                    #higher utilization means higher weight
                    utilization_weight = 1 + (link.get_utilization_percentage() / 100) * 10
                    G[u][v]['temp_weight'] = data['weight'] * utilization_weight
                else:
                    G[u][v]['temp_weight'] = data['weight']
            
            #find the path with the least total weight (considering utilization)
            path = nx.shortest_path(G, source, destination, weight='temp_weight')
            return path
        
        except nx.NetworkXNoPath:
            return []
        except Exception as e:
            print(f"Error finding least utilized path: {e}")
            return []
    
    def _find_load_balanced_path(self, source, destination):
        """Find a path for load balancing using weighted random selection"""
        try:
            k = 3
            paths = list(nx.shortest_simple_paths(self.topology, source, destination, weight='weight'))
            paths = paths[:k] if len(paths) > k else paths
            
            if not paths:
                return []
            
            if len(paths) == 1:
                return paths[0]
            
            #calculate weights for each path (inversely proportional to utilization)
            path_weights = []
            for path in paths:
                #calculate avg utilization along the path
                total_util = 0
                edge_count = 0
                
                for i in range(len(path) - 1):
                    link = self.links.get((path[i], path[i+1]))
                    if link:
                        total_util += link.get_utilization_percentage()
                        edge_count += 1
                
                avg_util = total_util / edge_count if edge_count > 0 else 0
                #higher utilization = lower weight (less likely to be chosen)
                weight = 100 - avg_util
                path_weights.append(max(1, weight))  # Ensure minimum weight of 1
            
            chosen_path = random.choices(paths, weights=path_weights, k=1)[0]
            return chosen_path
        
        except Exception as e:
            print(f"Error in load balancing: {e}")
            try:
                return nx.shortest_path(self.topology, source, destination, weight='weight')
            except:
                return []
    
    def add_flow(self, flow):
        """Add a new flow to the network and set up the required flow entries"""
        if not hasattr(flow, 'flow_id') or not flow.flow_id:
            self.flow_counter += 1
            flow.flow_id = f"flow_{self.flow_counter}"
        
        #compute path if not provided
        if not flow.path:
            flow.path = self.compute_path(flow.source, flow.destination, flow.traffic_type)
        
        if flow.path:
            self.flows[flow.flow_id] = flow
            self.stats['total_flows'] += 1
            self.stats['active_flows'] += 1
            
            self.setup_flow_path(flow)
            
            print(f"Added {flow}")
            return flow
        else:
            print(f"Cannot add flow from {flow.source} to {flow.destination}: No valid path")
            return None
    
    def setup_flow_path(self, flow):
        """Set up flow table entries along a path for a given flow"""
        if not flow.path or len(flow.path) < 2:
            return False
        
        #clear any existing flow entries for this flow
        self.clear_flow_entries(flow)
        
        #update links with flow information
        for i in range(len(flow.path) - 1):
            src, dst = flow.path[i], flow.path[i+1]
            link = self.links.get((src, dst))
            if link:
                link.add_flow(flow)
        
        #set up flow entries in each switch along the path
        for i in range(len(flow.path) - 1):
            switch_id = flow.path[i]
            next_hop = flow.path[i+1]
            
            if switch_id in self.switches:
                switch = self.switches[switch_id]
                
                output_port = switch.port_map.get(next_hop, 1)
                
                match_fields = {
                    'eth_type': 0x0800,  # IPv4
                    'ipv4_src': flow.source,
                    'ipv4_dst': flow.destination,
                    'flow_id': flow.flow_id
                }
                
                actions = [f"output:{output_port}"]
                
                priority = flow.traffic_type.value + 1
                
                entry = FlowTableEntry(match_fields, actions, priority)
                switch.add_flow_entry(entry)
        
        return True
    
    def clear_flow_entries(self, flow):
        """Remove all flow table entries for a specific flow"""
        for link in self.links.values():
            link.remove_flow(flow)
        
        for switch in self.switches.values():
            switch.remove_flow_entries({'flow_id': flow.flow_id})
    
    def remove_flow(self, flow_id):
        """Remove a flow from the network"""
        if flow_id in self.flows:
            flow = self.flows[flow_id]
            
            self.clear_flow_entries(flow)
            
            #update stats
            if flow.is_active:
                self.stats['active_flows'] -= 1
            
            #remove the flow
            del self.flows[flow_id]
            print(f"Removed flow {flow_id}")
            return True
        return False
    
    def simulate_link_failure(self, source, target):
        """Simulate a link failure event"""
        link_id = (source, target)
        if link_id in self.links:
            self.event_queue.put({'type': 'link_failure', 'link': link_id})
            return True
        return False
    
    def simulate_link_recovery(self, source, target):
        """Simulate a link recovery event"""
        link_id = (source, target)
        if source in self.switches and target in self.switches:
            self.event_queue.put({'type': 'link_recovery', 'link': link_id})
            return True
        return False
    
    def get_network_status(self):
        """Get current network status information"""
        return {
            'switches': len(self.switches),
            'links': len(self.links),
            'active_flows': self.stats['active_flows'],
            'total_flows': self.stats['total_flows'],
            'link_utilization': {
                f"{src} → {dst}": link.get_utilization_percentage()
                for (src, dst), link in self.links.items()
            }
        }
    
    def get_flow_path(self, flow_id):
        """Get the current path for a specific flow"""
        if flow_id in self.flows:
            flow = self.flows[flow_id]
            return flow.path
        return None
    
    def inject_traffic_flow(self, source, destination, traffic_type=TrafficType.NORMAL, 
                          bandwidth=1):
        """Inject a new traffic flow into the network"""
        self.flow_counter += 1
        flow_id = f"flow_{self.flow_counter}"
        
        #create the flow
        flow = Flow(flow_id, source, destination, traffic_type, bandwidth)
        
        #add it through the event queue
        self.event_queue.put({'type': 'new_flow', 'flow': flow})
        
        return flow_id
    
    def visualize_network(self, save_path=None):
        """Visualize the current network topology with link utilization"""
        plt.figure(figsize=(12, 8))
        
        #create positions for the nodes
        pos = nx.spring_layout(self.topology)
        
        #draw nodes (switches)
        nx.draw_networkx_nodes(self.topology, pos, node_size=700, node_color='lightblue')
        nx.draw_networkx_labels(self.topology, pos)
        
        #prepare edge colors based on utilization
        edge_colors = []
        edge_widths = []
        
        for u, v in self.topology.edges():
            link = self.links.get((u, v))
            if link:
                util_pct = link.get_utilization_percentage()
                
                #color mapping: green (0%) to red (100%)
                r = min(1.0, util_pct / 50)
                g = max(0.0, 1.0 - (util_pct / 50))
                edge_colors.append((r, g, 0))
                
                #width based on number of flows
                edge_widths.append(1 + 0.5 * len(link.flows))
            else:
                edge_colors.append('gray')
                edge_widths.append(1)
        
        #draw edges with appropriate colors
        nx.draw_networkx_edges(self.topology, pos, width=edge_widths, 
                              edge_color=edge_colors, arrows=True, 
                              arrowstyle='->', arrowsize=15)
        
        #edge labels with utilization info
        edge_labels = {}
        for u, v in self.topology.edges():
            link = self.links.get((u, v))
            if link:
                util = link.get_utilization_percentage()
                edge_labels[(u, v)] = f"{util:.1f}%"
        
        nx.draw_networkx_edge_labels(self.topology, pos, edge_labels=edge_labels, 
                                    font_size=8)
        
        plt.title("Network Topology with Link Utilization")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
        
        plt.tight_layout()
        plt.show()
    
    def close(self):
        """Clean up controller resources"""
        self.stop_event.set()
        self.event_thread.join(timeout=2)


class SDNControllerCLI(cmd.Cmd):
    """Command line interface for the SDN Controller"""
    intro = "Type 'help' to see commands."
    prompt = "SDN> "
    
    def __init__(self):
        super().__init__()
        self.controller = SDNController()
    
    def do_add_switch(self, arg):
        """Add a new switch: add_switch <switch_id>"""
        if not arg:
            print("Error: Switch ID is required")
            return
        
        self.controller.add_switch(arg)
    
    def do_remove_switch(self, arg):
        """Remove a switch: remove_switch <switch_id>"""
        if not arg:
            print("Error: Switch ID is required")
            return
        
        if self.controller.remove_switch(arg):
            print(f"Switch {arg} removed")
        else:
            print(f"Switch {arg} not found")
    
    def do_add_link(self, arg):
        """Add a link: add_link <source> <target> [capacity] [delay] [cost] [bidirectional]
        Example: add_link s1 s2 10 1 1 true"""
        args = arg.split()
        if len(args) < 2:
            print("Error: Source and target are required")
            return
        
        source, target = args[0], args[1]
        capacity = int(args[2]) if len(args) > 2 else 10
        delay = float(args[3]) if len(args) > 3 else 1
        cost = float(args[4]) if len(args) > 4 else 1
        bidirectional = args[5].lower() in ['true', 'yes', '1'] if len(args) > 5 else True
        
        self.controller.add_link(source, target, capacity, delay, cost, bidirectional)
    
    def do_remove_link(self, arg):
        """Remove a link: remove_link <source> <target> [bidirectional]
        Example: remove_link s1 s2 true"""
        args = arg.split()
        if len(args) < 2:
            print("Error: Source and target are required")
            return
        
        source, target = args[0], args[1]
        bidirectional = args[2].lower() in ['true', 'yes', '1'] if len(args) > 2 else True
        
        if self.controller.remove_link(source, target, bidirectional):
            print(f"Link {source} → {target} removed")
        else:
            print(f"Link {source} → {target} not found")
    
    def do_add_flow(self, arg):
        """Add a flow: add_flow <source> <destination> <traffic_type> [bandwidth]
        Traffic types: CRITICAL, BUSINESS, NORMAL, BACKGROUND
        Example: add_flow s1 s5 BUSINESS 2"""
        args = arg.split()
        if len(args) < 3:
            print("Error: Source, destination, and traffic type are required")
            return
        
        source, destination, traffic_type_str = args[0], args[1], args[2]
        
        try:
            traffic_type = TrafficType[traffic_type_str.upper()]
        except KeyError:
            print(f"Error: Invalid traffic type. Valid types are: {', '.join(t.name for t in TrafficType)}")
            return
        
        bandwidth = float(args[3]) if len(args) > 3 else 1
        
        flow_id = self.controller.inject_traffic_flow(source, destination, traffic_type, bandwidth)
        print(f"Flow {flow_id} added")
    
    def do_remove_flow(self, arg):
        """Remove a flow: remove_flow <flow_id>"""
        if not arg:
            print("Error: Flow ID is required")
            return
        
        if self.controller.remove_flow(arg):
            print(f"Flow {arg} removed")
        else:
            print(f"Flow {arg} not found")
    
    def do_fail_link(self, arg):
        """Simulate a link failure: fail_link <source> <target>"""
        args = arg.split()
        if len(args) < 2:
            print("Error: Source and target are required")
            return
        
        source, target = args[0], args[1]
        if self.controller.simulate_link_failure(source, target):
            print(f"Link failure simulated: {source} → {target}")
        else:
            print(f"Link {source} → {target} not found")
    
    def do_recover_link(self, arg):
        """Simulate a link recovery: recover_link <source> <target>"""
        args = arg.split()
        if len(args) < 2:
            print("Error: Source and target are required")
            return
        
        source, target = args[0], args[1]
        if self.controller.simulate_link_recovery(source, target):
            print(f"Link recovery simulated: {source} → {target}")
        else:
            print(f"Cannot recover link: {source} or {target} not found")
    
    def do_show_topology(self, arg):
        """Visualize the network topology: show_topology [save_path]"""
        save_path = arg if arg else None
        self.controller.visualize_network(save_path)
    
    def do_show_network_status(self, arg):
        """Show current network status"""
        status = self.controller.get_network_status()
        print("Network Status:")
        print(f"  Switches: {status['switches']}")
        print(f"  Links: {status['links']}")
        print(f"  Active Flows: {status['active_flows']}")
        print(f"  Total Flows: {status['total_flows']}")
        print("Link Utilization:")
        for link, util in status['link_utilization'].items():
            print(f"  {link}: {util:.2f}%")
    
    def do_show_flows(self, arg):
        """Show all active flows"""
        print("Active Flows:")
        for flow_id, flow in self.controller.flows.items():
            if flow.is_active:
                path_str = " → ".join(flow.path)
                print(f"  {flow_id}: {flow.source} → {flow.destination} ({flow.traffic_type.name}, {flow.bandwidth} Gbps)")
                print(f"    Path: {path_str}")
    
    def do_show_flow(self, arg):
        """Show details of a specific flow: show_flow <flow_id>"""
        if not arg:
            print("Error: Flow ID is required")
            return
        
        flow = self.controller.flows.get(arg)
        if flow:
            print(f"Flow {flow.flow_id}:")
            print(f"  Source: {flow.source}")
            print(f"  Destination: {flow.destination}")
            print(f"  Traffic Type: {flow.traffic_type.name}")
            print(f"  Bandwidth: {flow.bandwidth} Gbps")
            print(f"  Active: {flow.is_active}")
            print(f"  Path: {' → '.join(flow.path)}")
        else:
            print(f"Flow {arg} not found")
    
    def do_show_switch(self, arg):
        """Show details of a specific switch: show_switch <switch_id>"""
        if not arg:
            print("Error: Switch ID is required")
            return
        
        switch = self.controller.switches.get(arg)
        if switch:
            print(f"Switch {switch.switch_id}:")
            print(f"  Connected Links: {', '.join(str(link) for link in switch.connected_links)}")
            print(f"  Port Map: {switch.port_map}")
            print(f"  Flow Table ({len(switch.flow_table)} entries):")
            for i, entry in enumerate(switch.flow_table):
                print(f"    {i+1}. {entry}")
        else:
            print(f"Switch {arg} not found")
    
    def do_create_test_network(self, arg):
        """Create a test network with predefined topology"""
        #clear existing network
        for flow_id in list(self.controller.flows.keys()):
            self.controller.remove_flow(flow_id)
        
        for switch_id in list(self.controller.switches.keys()):
            self.controller.remove_switch(switch_id)
        
        #create switches
        switches = ['s1', 's2', 's3', 's4', 's5']
        for s in switches:
            self.controller.add_switch(s)
        
        #create links (forming a simple mesh)
        self.controller.add_link('s1', 's2', 10, 1, 1)
        self.controller.add_link('s1', 's3', 5, 2, 2)
        self.controller.add_link('s2', 's4', 8, 1, 1)
        self.controller.add_link('s3', 's4', 10, 1, 1)
        self.controller.add_link('s2', 's5', 5, 2, 2)
        self.controller.add_link('s4', 's5', 10, 1, 1)
        
        print("Test network created")
    
    def do_create_test_flows(self, arg):
        """Create some test flows in the network"""
        flows = [
            ('s1', 's5', TrafficType.CRITICAL, 2),
            ('s1', 's4', TrafficType.BUSINESS, 1),
            ('s2', 's3', TrafficType.NORMAL, 1),
            ('s3', 's5', TrafficType.BACKGROUND, 0.5)
        ]
        
        for src, dst, t_type, bw in flows:
            self.controller.inject_traffic_flow(src, dst, t_type, bw)
        
        print("Test flows created")
    
    def do_save_topology(self, arg):
        """Save the current topology to a JSON file: save_topology <filename>"""
        if not arg:
            print("Error: Filename is required")
            return
        
        try:
            data = {
                'switches': list(self.controller.switches.keys()),
                'links': [
                    {
                        'source': src,
                        'target': dst,
                        'capacity': link.capacity,
                        'delay': link.delay,
                        'cost': link.cost
                    } for (src, dst), link in self.controller.links.items()
                ]
            }
            
            with open(arg, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Topology saved to {arg}")
        except Exception as e:
            print(f"Error saving topology: {e}")
    
    def do_load_topology(self, arg):
        """Load a topology from a JSON file: load_topology <filename>"""
        if not arg:
            print("Error: Filename is required")
            return
        
        try:
            with open(arg, 'r') as f:
                data = json.load(f)
            
            #clear existing network
            for flow_id in list(self.controller.flows.keys()):
                self.controller.remove_flow(flow_id)
            
            for switch_id in list(self.controller.switches.keys()):
                self.controller.remove_switch(switch_id)
            
            #create switches
            for switch_id in data.get('switches', []):
                self.controller.add_switch(switch_id)
            
            #create links
            for link_data in data.get('links', []):
                self.controller.add_link(
                    link_data['source'],
                    link_data['target'],
                    link_data.get('capacity', 10),
                    link_data.get('delay', 1),
                    link_data.get('cost', 1)
                )
            
            print(f"Topology loaded from {arg}")
        except Exception as e:
            print(f"Error loading topology: {e}")
    
    def do_exit(self, arg):
        """Exit the SDN Controller CLI"""
        print("Cleaning up...")
        self.controller.close()
        print("Goodbye!")
        return True
    
    def do_quit(self, arg):
        """Exit the SDN Controller CLI"""
        return self.do_exit(arg)


def main():
    """Main function to start the SDN controller CLI"""
    cli = SDNControllerCLI()
    
    #check for command line arguments
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            print("Creating test network...")
            cli.do_create_test_network("")
            cli.do_create_test_flows("")
    
    try:
        cli.cmdloop()
    except KeyboardInterrupt:
        print("\nInterrupted")
        cli.do_exit("")
    except Exception as e:
        print(f"Error: {e}")
        cli.do_exit("")


if __name__ == "__main__":
    main()