# this script is responsible to acquire and label the data according to each scenario. It requires the scenario information in
# terms of 'scenarios.csv' as input inside the 'grid_topology' folder. For some other auxiliary functions the script will
# need to access the 'buses.csv' and 'lines.csv'


import numpy as np
from joblib import dump

class DataCollector:

    def __init__(self,cwd_path):
        self.cwd_path = cwd_path
        self.status = "Collecting data from disk"
        print(self.status)

    def __transform_to_polar__(self,cartesian_input):
        rows = cartesian_input.shape[0]
        cols = 2
        polar_output = np.zeros([rows,cols])
        polar_output[:,0] = np.abs(cartesian_input[:, 0] + cartesian_input[:, 1] * 1j) #magnitude
        buff = np.angle(cartesian_input[:, 0] + cartesian_input[:, 1] * 1j) #angle in radians
        # checking for negative angles, if present convert them to anti-clockwise measurement
        buff[np.where(buff<0)] = 2*np.pi+buff[np.where(buff<0)]
        polar_output[:,1] = buff

        return polar_output

    def __getNodesConnected__(self,node,lines,faulty_line):
        conn_nodes = []

        for i in range(lines.shape[0]):
            if node == lines[i,1] and lines[i,0] != faulty_line and lines[i,3] != 0:
                conn_nodes.append(lines[i,2])
            if node == lines[i,2] and lines[i,0] != faulty_line and lines[i,3] != 0:
                conn_nodes.append(lines[i,1])


        return conn_nodes

    def __getAffectedNodesLines__(self,fb, tb, nodes, lines):
        #this function is an implementation of an algorithm that tries to find out which nodes and lines are (un)affected.
        lines_affected = []
        nodes_affected = []
        lines_not_affected = []
        nodes_not_affected = []



        source_node = nodes[0,1]
        for i in range(nodes.shape[0]):
            if nodes[i,2] == 1:
                source_node = nodes[i,1]

        connected_nodes = [source_node]

        faulty_line = 0

        for i in range(lines.shape[0]):
            if (lines[i,1] == fb and lines[i,2] == tb) or (lines[i,1] == tb and lines[i,2] == fb):
                faulty_line = lines[i,0]




        if faulty_line != 0:
            for i in range(lines.shape[0]):
                if lines[i,1] == source_node and lines[i,0] != faulty_line and lines[i,3] != 0:
                    connected_nodes.append(lines[i,2])
                if lines[i,2] == source_node and lines[i,0] != faulty_line and lines[i,3] != 0:
                    connected_nodes.append(lines[i,1])

            flg = 1
            while flg:
                l1 = len(connected_nodes)
                for node1 in connected_nodes:
                    new_connected_nodes = self.__getNodesConnected__(node1,lines,faulty_line)
                    if len(new_connected_nodes):
                        for node2 in new_connected_nodes:
                            if node2 not in connected_nodes:
                                connected_nodes.append(node2)
                l2 = len(connected_nodes)
                if l2==l1:
                    flg=0
        else:
            connected_nodes = []
            for i in range(nodes.shape[0]):
                if nodes[i,2] != 0:
                    connected_nodes.append(int(nodes[i,1]))



        nodes_not_affected.extend(connected_nodes)

        for i in range(nodes.shape[0]):
            if nodes[i,1] not in nodes_not_affected and nodes[i,2] != 0:
                nodes_affected.append(nodes[i,1])



        for i in range(lines.shape[0]):
            if lines[i,3] != 0:
                if (lines[i,1] in nodes_not_affected and lines[i,2] in nodes_not_affected) or (lines[i,2] in nodes_not_affected and lines[i,1] in nodes_not_affected) and lines[i,0] not in lines_not_affected:
                    lines_not_affected.append(lines[i,0])
                else:
                    lines_affected.append(lines[i,0])


        return lines_affected, nodes_affected, lines_not_affected, nodes_not_affected

    def __getNodeVoltageData__(self,nodes_affected, nodes_not_affected, cwd_path,key,transform_cartesian_to_polar,total_number_of_obs,obs_number):
        #   this function gathers the measurements saved after the powerflow simulation and aggregates them as nodal data
        #   e.g. voltages of all the nodes for all three phases in rectangular/polar form pertaining to each scenario
        scenario_dir = (str(cwd_path) + "/scenario_data/" + str(key))
        total_number_of_samples = 14399  # number of samples generated ##TODO: remove this variable

        number_of_samples = int(total_number_of_samples / total_number_of_obs)

        rows = number_of_samples


        begin_idx = int(number_of_samples * (obs_number - 1))
        end_idx = int(number_of_samples * obs_number - 1)


        cols = 2 #only considering single-phase measurements - real and imaginary
        k = 1 #for phase A measurements, 3 for phase B & 5 for phase C
        node_ids = []

        if len(nodes_not_affected) == 1:
            filename = scenario_dir+"/node" + str(int(nodes_not_affected[0])) + ".csv"
            if transform_cartesian_to_polar:
                node_data = self.__transform_to_polar__(np.genfromtxt(filename,delimiter=',')[begin_idx:end_idx+1, k:cols+k])
            else:
                node_data = np.genfromtxt(filename,delimiter=',')[begin_idx:end_idx-1, k:cols+k]
        elif len(nodes_not_affected) > 1:
            flg = 0
            for node in nodes_not_affected:
                filename = scenario_dir+"/node" + str(int(node)) + ".csv"
                if flg == 0:
                    if transform_cartesian_to_polar:
                        node_data = self.__transform_to_polar__(np.genfromtxt(filename, delimiter=',')[begin_idx:end_idx+1, k:cols + k])
                    else:
                        node_data = np.genfromtxt(filename,delimiter=',')[begin_idx:end_idx-1, k:cols+k]
                    flg = 1
                else:
                    if transform_cartesian_to_polar:
                        buff = self.__transform_to_polar__(np.genfromtxt(filename, delimiter=',')[begin_idx:end_idx+1, k:cols + k])
                    else:
                        buff = np.genfromtxt(filename,delimiter=',')[begin_idx:end_idx+1, k:cols+k]
                    node_data = np.hstack((node_data,buff))


        if len(nodes_affected) == 1:
            buff = np.zeros([rows,cols])
            node_data = np.hstack((node_data, buff))
        elif len(nodes_affected) > 1:
            for node in nodes_affected:
                buff = np.zeros([rows, cols])
                node_data = np.hstack((node_data,buff))

        ## this is done only if single phase measurements are considered
        if transform_cartesian_to_polar==False and cols == 2:
            node_data = np.abs(node_data)

        if transform_cartesian_to_polar:
            for node in nodes_not_affected:
                node_ids.append("n" + str(node) + "_voltage_A_mag")
                node_ids.append("n" + str(node) + "_voltage_A_ang")
            for node in nodes_affected:
                node_ids.append("n" + str(node) + "_voltage_A_mag")
                node_ids.append("n" + str(node) + "_voltage_A_ang")
        else:
            for node in nodes_not_affected:
                node_ids.append("n" + str(node) + "_voltage_A_real")
                node_ids.append("n" + str(node) + "_voltage_A_imag")
            for node in nodes_affected:
                node_ids.append("n" + str(node) + "_voltage_A_real")
                node_ids.append("n" + str(node) + "_voltage_A_imag")


        return node_data, node_ids

    def __getLineData__(self,lines_affected, lines_not_affected,cwd_path,key,transform_cartesian_to_polar,total_number_of_obs,obs_number):
        scenario_dir = (str(cwd_path) + "/scenario_data/" + str(key))
        total_number_of_samples = 14399  # number of samples generated ##TODO: remove this variable

        number_of_samples = int(total_number_of_samples / total_number_of_obs)

        rows = number_of_samples

        begin_idx = int(number_of_samples * (obs_number - 1))
        end_idx = int(number_of_samples * obs_number - 1)


        cols = 2 # 1 for only considering phase A active power flow, 2 for both active and reactive power flow
        k = 1  # for phase A measurements, 3 for phase B & 5 for phase C
        line_ids = []

        if len(lines_not_affected) == 1:
            filename = scenario_dir+"/line" + str(int(lines_not_affected[0])) + ".csv"
            if transform_cartesian_to_polar:
                line_data = self.__transform_to_polar__(np.genfromtxt(filename,delimiter=',')[begin_idx:end_idx+1, k:cols+k])
            else:
                line_data = np.genfromtxt(filename,delimiter=',')[begin_idx:end_idx-1, k:cols+k]
        elif len(lines_not_affected) > 1:
            flg = 0
            for line in lines_not_affected:
                filename = scenario_dir+"/line" + str(int(line)) + ".csv"
                if flg == 0:
                    if transform_cartesian_to_polar:
                        line_data = self.__transform_to_polar__(np.genfromtxt(filename, delimiter=',')[begin_idx:end_idx+1, k:cols + k])
                    else:
                        line_data = np.genfromtxt(filename, delimiter=',')[begin_idx:end_idx + 1, k:cols + k]
                    flg = 1
                else:
                    if transform_cartesian_to_polar:
                        buff = self.__transform_to_polar__(np.genfromtxt(filename, delimiter=',')[begin_idx:end_idx+1, k:cols + k])
                    else:
                        buff = np.genfromtxt(filename,delimiter=',')[begin_idx:end_idx+1, k:cols+k]
                    line_data = np.hstack((line_data, buff))


        if len(lines_not_affected) == 0:
            flg = 0
            for line in lines_affected:
                if flg == 0:
                    line_data = np.zeros([rows,cols])
                    flg = 1
                else:
                    buff = np.zeros([rows,cols])
                    line_data = np.hstack((line_data,buff))
        else:
            if len(lines_affected) == 1:
                buff = np.zeros([rows,cols])
                line_data = np.hstack((line_data, buff))
            elif len(lines_affected) > 1:
                for line in lines_affected:
                    buff = np.zeros([rows, cols])
                    line_data = np.hstack((line_data,buff))

        ## convert all power flow measurements to positive if single phase measurement is considered
        if cols==1:
            line_data = np.abs(line_data)

        ##TODO: if double-phase or three-phase measurements are considered, please change the powerflow to polar form


        for line in lines_not_affected:
            line_ids.append("l"+str(int(line)) + "_power_in_A_real")
            line_ids.append("l" + str(int(line)) + "_power_in_A_imag")
        for line in lines_affected:
            line_ids.append("l"+str(int(line)) + "_power_in_A_real")
            line_ids.append("l" + str(int(line)) + "_power_in_A_imag")

        return line_data, line_ids

    def __getScenarioData__(self,transform_cartesian_to_polar=True,total_number_of_obs=1,obs_number=1):

        scenarios = np.genfromtxt(str(self.cwd_path)+"/scenario_data/scenarios.csv",delimiter=',')
        lines = np.genfromtxt(str(self.cwd_path) + "/grid_topology/lines.csv", delimiter=',')
        nodes = np.genfromtxt(str(self.cwd_path) + "/grid_topology/buses.csv", delimiter=',')
        node_voltage_data = {}
        line_data = {}
        for i in range(scenarios.shape[0]):
            lines_affected, nodes_affected, lines_not_affected, nodes_not_affected = self.__getAffectedNodesLines__(scenarios[i,1],scenarios[i,2],nodes,lines)
            key = str(int(scenarios[i,1]))+"_"+str(int(scenarios[i,2]))

            node_voltage_data[key], node_ids = self.__getNodeVoltageData__(nodes_affected, nodes_not_affected,
                                                                           self.cwd_path, key, transform_cartesian_to_polar,
                                                                           total_number_of_obs, obs_number)

            line_data[key], line_ids = self.__getLineData__(lines_affected, lines_not_affected, self.cwd_path,
                                                            key, transform_cartesian_to_polar,
                                                            total_number_of_obs, obs_number)

        node_list = []
        for i in range(nodes.shape[0]):
            if nodes[i,2] != 0:
                node_list.append(int(nodes[i, 1]))

        line_dict = {}
        for i in range(lines.shape[0]):
            line_dict[int(lines[i,0])] = [int(lines[i, 1]), int(lines[i, 2]), int(lines[i,3])]

        return {"node_voltage_data":node_voltage_data, "node_ids":node_ids, "line_data":line_data,
                "line_ids":line_ids, "line_dict":line_dict, "node_list":node_list}



    def get_all_data(self):
        all_scenario_data = self.__getScenarioData__()

        node_data = all_scenario_data['node_voltage_data']
        line_data = all_scenario_data['line_data']
        flg = 0
        for scenario in node_data:
            if flg == 0:
                input_data = np.hstack((node_data[scenario], line_data[scenario]))
                output_data = np.array([str(scenario) for _ in range(input_data.shape[0])]).reshape(-1,1)
                flg = 1
            else:
                buff_in = np.hstack((node_data[scenario], line_data[scenario]))
                buff_out = np.array([str(scenario) for _ in range(buff_in.shape[0])]).reshape(-1,1)
                input_data = np.vstack((input_data, buff_in))
                output_data = np.vstack((output_data, buff_out))

        all_data = {"input_data" : input_data,
                    "output_data": output_data,
                    "node_ids"   : all_scenario_data['node_ids'],
                    "line_ids"   : all_scenario_data['line_ids'],
                    "line_dict"  : all_scenario_data['line_dict'],
                    "node_list"  : all_scenario_data['node_list']}

        dump(all_data, self.cwd_path + "/scenario_data/all_data.pkl")
        self.status = "Data collected and dumped back to disk"
        print(self.status)

        return None



