from copy import deepcopy
import numpy as np
from src.create_grid_object import Grid


class GridSplitter:

    def __init__(self,grid_obj,parent_grid_obj=None):
        self.grid = deepcopy(grid_obj)

        if parent_grid_obj == None:
            self.parent_grid = grid_obj
        else:
            self.parent_grid = parent_grid_obj

        self.status = "Making sub-zones"
        print(self.status)

    def __do_search__(self,grid,start_node,list_of_branches,flg,algorithm="BFS"):


        if algorithm=="BFS":
            ############## BFS algorithm ##############
            # print("executing BFS..")
            new_start_nodes = []
            old_start_node = start_node
            for line in grid.line_list:
                if line not in list_of_branches:
                    if old_start_node == line[0]:
                        list_of_branches.append(line)
                        new_start_nodes.append(line[1])
                    elif old_start_node == line[1]:
                        list_of_branches.append(line)
                        new_start_nodes.append(line[0])
                    else:
                        flg = 0

            for node in new_start_nodes:
                list_of_branches, flg = self.__do_search__(grid,node,list_of_branches, flg)
        else:
            ############## DFS algorithm ##############
            # print("executing DFS..")
            old_start_node = start_node
            for line in self.grid.line_list:
                if line not in list_of_branches:
                    if old_start_node == line[0]:
                        list_of_branches.append(line)
                        new_start_nodes = (line[1])
                        list_of_branches, flg = self.__do_search__(grid,new_start_nodes, list_of_branches, flg,algorithm="DFS")
                    elif old_start_node == line[1]:
                        list_of_branches.append(line)
                        new_start_nodes = (line[0])
                        list_of_branches, flg = self.__do_search__(grid,new_start_nodes, list_of_branches, flg,algorithm="DFS")
                    else:
                        flg = 0


        return list_of_branches, flg

    def __get_number_of_branches__(self,grid,node_number,line):
        list_of_branches = []
        if node_number == line[0]:
            fb = node_number
            tb = line[1]
        else:
            fb = node_number
            tb = line[0]

        list_of_branches.append(line)

        fb = tb

        list_of_branches, _ = self.__do_search__(grid,fb, list_of_branches, flg=1)

        number_of_branches = len(list_of_branches)

        return number_of_branches, list_of_branches

    def __get_list_of_number_of_branches__(self,grid,node_number):
        list_of_number_of_branches = []
        list_of_conn_branches = []

        list_of_lines = grid.node_dict[node_number]
        for line_nr in list_of_lines:
            #if grid.line_dict[line_nr][2] != 0:
            line = [grid.line_dict[line_nr][0], grid.line_dict[line_nr][1]]
            number_of_branches, _ = self.__get_number_of_branches__(grid,node_number, line)
            list_of_number_of_branches.append(number_of_branches)
            list_of_conn_branches.append([line[0], line[1]])

        return list_of_number_of_branches, list_of_conn_branches

    def __objective_func__(self,x,list_of_number_of_branches,number_of_parts):
        x_ = list(np.binary_repr(x,len(list_of_number_of_branches)))
        y = 0
        for bit,ele in zip(x_,list_of_number_of_branches):
            y = y + int(bit)*ele

        approx_number_of_lines_in_each_child_grid = (len(self.grid.line_list) / number_of_parts)


        y = np.abs(y - approx_number_of_lines_in_each_child_grid)

        return y

    def __find_best_score__(self,list_of_number_of_branches,number_of_parts):
        result = []


        list_of_combinations = list(np.arange(2**len(list_of_number_of_branches)))
        for comb in list_of_combinations:
            result.append(self.__objective_func__(comb,list_of_number_of_branches,number_of_parts))

        result_with_combinations_unsorted = np.vstack((result,list_of_combinations))

        result_with_combinations_sorted = result_with_combinations_unsorted[:,result_with_combinations_unsorted[0,:].argsort()]

        best_score = result_with_combinations_sorted[0,0]
        best_combination = list(np.binary_repr(int(result_with_combinations_sorted[1, 0]),len(list_of_number_of_branches)))

        return best_score, best_combination

    def __scoring_function__(self,grid,node_number,number_of_parts):

        list_of_number_of_branches,_ = self.__get_list_of_number_of_branches__(grid,node_number)

        value, combination = self.__find_best_score__(list_of_number_of_branches,number_of_parts)
        combination_of_lines = []

        for bit, line_nr in zip(combination, grid.node_dict[node_number]):
            if int(bit) != 0:
                combination_of_lines.append(line_nr)

        return value, combination_of_lines

    def __get_child_grid__(self,parent_grid,optimal_location,current_optimal_node,child_number=0):
        ### creating empty child grids
        child_grid_A = Grid(self.grid)
        child_grid_B = Grid(self.grid)

        ### populating grid ids

        child_grid_A.grid_id = child_grid_A.parent_grid_id + str(child_number)
        child_grid_B.grid_id = child_grid_B.parent_grid_id + str(child_number+1)





        ### populating node and line list for child grid A
        for optimal_node in optimal_location:
            node_list = []
            line_list = []
            for optimal_line in optimal_location[optimal_node]:
                line_list.extend(self.__get_number_of_branches__(parent_grid,optimal_node, optimal_line)[1])
            for line in line_list:
                if line[0] not in node_list:
                    node_list.append(line[0])
                if line[1] not in node_list:
                    node_list.append(line[1])
        child_grid_A.node_list = list(node_list)
        child_grid_A.line_list = list(line_list)

        ### separating common nodes to be later appended in child grid B
        common_nodes = [current_optimal_node]  ################

        ### populating node list for child grid B
        child_grid_B.node_list = list(set(parent_grid.node_list)-set(child_grid_A.node_list)-set(common_nodes))

        ### this is the point where the optimal nodes are put in both grids
        child_grid_B.node_list.extend(common_nodes)

        ### populating line list for child grid B

        for line in parent_grid.line_list:
            if line not in child_grid_A.line_list:
                child_grid_B.line_list.append(line)


        ### populating line dicts for child grid A & B
        for line_nr in parent_grid.line_dict:
            #if parent_grid.line_dict[line_nr][2] != 0:
            line = [parent_grid.line_dict[line_nr][0], parent_grid.line_dict[line_nr][1]]
            if line in child_grid_A.line_list:
                child_grid_A.line_dict[line_nr] = [parent_grid.line_dict[line_nr][0],
                                                   parent_grid.line_dict[line_nr][1],
                                                   parent_grid.line_dict[line_nr][2]]
            else:
                child_grid_B.line_dict[line_nr] = [parent_grid.line_dict[line_nr][0],
                                                   parent_grid.line_dict[line_nr][1],
                                                   parent_grid.line_dict[line_nr][2]]

        ### populating node dict for child grid A
        for node in child_grid_A.node_list:
            child_grid_A.node_dict[node] = []
            for line in child_grid_A.line_dict:
                fb = child_grid_A.line_dict[line][0]
                tb = child_grid_A.line_dict[line][1]
                if fb == node or tb == node and line not in child_grid_A.node_dict[node]:
                    child_grid_A.node_dict[node].append(line)

        ### populating node dict for child grid B
        for node in child_grid_B.node_list:
            child_grid_B.node_dict[node] = []
            for line in child_grid_B.line_dict:
                fb = child_grid_B.line_dict[line][0]
                tb = child_grid_B.line_dict[line][1]
                if fb == node or tb == node and line not in child_grid_B.node_dict[node]:
                    child_grid_B.node_dict[node].append(line)

        ### populating grid-type for child grid A
        if len(child_grid_A.line_list) == len(child_grid_A.node_list) - 1:
            child_grid_A.grid_type = "radial"
        else:
            child_grid_A.grid_type = "non-radial"

        ### populating grid-type for child grid B
        if len(child_grid_B.line_list) == len(child_grid_B.node_list) - 1:
            child_grid_B.grid_type = "radial"
        else:
            child_grid_B.grid_type = "non-radial"



        #### populating child grid properties from the main parent grid
        ### populating child grid feature ids and inds


        child_grid_A.node_feature_ids = list(self.parent_grid.node_feature_ids)
        child_grid_A.line_feature_ids = list(self.parent_grid.line_feature_ids)
        child_grid_A.feature_ids = list(self.parent_grid.feature_ids)
        child_grid_A.feature_inds = list(self.parent_grid.feature_inds)
        child_grid_B.node_feature_ids = list(self.parent_grid.node_feature_ids)
        child_grid_B.line_feature_ids = list(self.parent_grid.line_feature_ids)
        child_grid_B.feature_ids = list(self.parent_grid.feature_ids)
        child_grid_B.feature_inds = list(self.parent_grid.feature_inds)

        ### populating total classification labels dict for child grid A & B
        for line in self.parent_grid.line_list:
            key = str(line[0]) + "_" + str(line[1])
            if key in self.parent_grid.total_clf_labels_dict:
                #print(key)
                if line in child_grid_A.line_list:
                    child_grid_A.total_clf_labels_dict[key] = key
                else:
                    child_grid_A.total_clf_labels_dict[key] = "0_0"
                if line in child_grid_B.line_list:
                    child_grid_B.total_clf_labels_dict[key] = key
                else:
                    child_grid_B.total_clf_labels_dict[key] = "0_0"
        child_grid_A.total_clf_labels_dict["0_0"] = "0_0"
        child_grid_B.total_clf_labels_dict["0_0"] = "0_0"





        return child_grid_A, child_grid_B

    def __check_splitting_consistency__(self,parent_grid,list_of_child_grids,optimal_locations, number_of_parts):

        all_lines = []
        for grid in list_of_child_grids:
            if grid.line_list not in all_lines:
                all_lines.extend(grid.line_list)

        if len(all_lines) != len(parent_grid.line_list):
            flg = False
        else:
            flg = True
            print(optimal_locations)

        return flg

    def __split_grid__(self,number_of_parts=2,split_method="exponential"):

        list_of_child_grids = []
        optimal_locations = {}
        if split_method == "exponential":
            #### brute-force method to compute the objective function value for all nodes and then find the minimum
            score=[]
            for node in self.grid.node_list:
                score.append(self.__scoring_function__(self.grid,node, number_of_parts))

            node_with_scores_unsorted = np.vstack((self.grid.node_list,score))
            node_with_scores_sorted = node_with_scores_unsorted[:,(node_with_scores_unsorted[1,:]).argsort()]
            optimal_node = int(node_with_scores_sorted[0,0])


            optimal_locations[optimal_node] = []
            number_of_branches, lines = self.__get_list_of_number_of_branches__(self.grid,optimal_node)
            optimal_locations[optimal_node].append(lines[number_of_branches.index(max(number_of_branches))])
        else:
            optimal_locations = []
            new_parent_grid = deepcopy(self.grid)
            '''
            if number_of_parts != 2:
                if len(new_parent_grid.line_list) % number_of_parts == 0:
                    n_iter = number_of_parts-1
                else:
                    n_iter = number_of_parts
            else:
                n_iter = number_of_parts-1
                
            for i in range(n_iter):
                optimal_node_dict = {}
                score = []
                combination_dict = {}
                for node in new_parent_grid.node_list:
                    res = self.__scoring_function__(new_parent_grid,node, number_of_parts)
                    score.append(res[0])
                    combination_dict[node] = res[1]

                node_with_scores_unsorted = np.vstack((new_parent_grid.node_list, score))
                node_with_scores_sorted = node_with_scores_unsorted[:, (node_with_scores_unsorted[1, :]).argsort()]
                optimal_node = int(node_with_scores_sorted[0,0])
                optimal_node_dict[optimal_node] = []

                for line_nr in combination_dict[optimal_node]:
                    optimal_node_dict[optimal_node].append([self.grid.line_dict[line_nr][0], self.grid.line_dict[line_nr][1]])
                child_grids = self.__get_child_grid__(new_parent_grid, optimal_node_dict, optimal_node,child_number=i+1)
                list_of_child_grids.append(child_grids[0])
                self.status = "Child grid created with grid-id: " + list_of_child_grids[-1].grid_id
                print(self.status)
                if i == n_iter-1:
                    list_of_child_grids.append(child_grids[1])
                    self.status = "Child grid created with grid-id: " + list_of_child_grids[-1].grid_id
                    print(self.status)
                else:
                    del new_parent_grid
                    new_parent_grid = deepcopy(child_grids[1])
                    del child_grids
                optimal_locations.append(optimal_node_dict)
            '''
            flg = True
            i = 0
            while flg:
                optimal_node_dict = {}
                score = []
                combination_dict = {}
                for node in new_parent_grid.node_list:
                    res = self.__scoring_function__(new_parent_grid, node, number_of_parts)
                    score.append(res[0])
                    combination_dict[node] = res[1]

                node_with_scores_unsorted = np.vstack((new_parent_grid.node_list, score))
                node_with_scores_sorted = node_with_scores_unsorted[:, (node_with_scores_unsorted[1, :]).argsort()]
                optimal_node = int(node_with_scores_sorted[0, 0])
                optimal_node_dict[optimal_node] = []

                for line_nr in combination_dict[optimal_node]:
                    optimal_node_dict[optimal_node].append(
                        [self.grid.line_dict[line_nr][0], self.grid.line_dict[line_nr][1]])
                child_grids = self.__get_child_grid__(new_parent_grid, optimal_node_dict, optimal_node,
                                                      child_number=i + 1)
                list_of_child_grids.append(child_grids[0])
                self.status = "Child grid created with grid-id: " + list_of_child_grids[-1].grid_id
                print(self.status)
                if len(child_grids[1].line_list)<=len(child_grids[0].line_list):
                    if len(child_grids[1].line_list) > 0:
                        list_of_child_grids.append(child_grids[1])
                        self.status = "Child grid created with grid-id: " + list_of_child_grids[-1].grid_id
                        print(self.status)
                    flg = False
                else:
                    del new_parent_grid
                    new_parent_grid = deepcopy(child_grids[1])
                    del child_grids
                    i = i+1
                optimal_locations.append(optimal_node_dict)

        flg = self.__check_splitting_consistency__(self.grid,list_of_child_grids,optimal_locations,number_of_parts)


        if flg:
            return list_of_child_grids, flg
        else:
            print("Splitting inconsistent!")
            return [], flg




    def make_sub_zones(self,number_of_parts=2,split_method="exponential"):

        if self.grid.grid_type == "radial":
            list_of_child_grids, flg = self.__split_grid__(number_of_parts,split_method)
        else:
            ## TODO : separate strategy to make child grids for non-radial grids
            list_of_child_grids = []
            pass


        return list_of_child_grids, self.grid, flg