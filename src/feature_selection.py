import numpy as np
from sklearn.feature_selection import mutual_info_classif
from src.create_grid_object import Grid
from copy import deepcopy

class FeatureSorter:

    def __init__(self, parent_grid_obj: Grid, child_grid_obj:Grid = None):
        self.grid = deepcopy(parent_grid_obj)
        self.child_grid = deepcopy(child_grid_obj)
        self.status = "Sorting grid features"
        print(self.status)

    def __getConnectedLines__(self,node,lines):
        conn_lines = []
        '''
        for i in range(lines.shape[0]):
            if (node == lines[i,1] or node == lines[i,2]) and lines[i,3] != 0:
                conn_lines.append(int(lines[i,0]))
        '''
        for line_nr in lines:
            val = lines[line_nr]
            if (node == val[1] or node == val[0]) and val[2] != 0:
                conn_lines.append(line_nr)
        return conn_lines

    def __getConnectedNodes__(self,line_number, lines):
        """
        if lines[line_number-1,3] != 0:
            conn_nodes = [int(lines[line_number-1,1]),int(lines[line_number-1,2])]
        else:
            conn_nodes = []
        """
        if lines[line_number][2] != 0:
            conn_nodes = [lines[line_number][0],lines[line_number][1]]
        else:
            conn_nodes = []

        return conn_nodes

    def __getFeatureIndLines__(self,feature_id_sorted,lines,feature_ind_sorted):
        feature_ind_of_selected_line = len(feature_id_sorted)-1

        for line in lines:
            for id in feature_id_sorted:
                if id[0] == "l":
                    line_number = int(''.join(list(filter(str.isdigit, id))))
                    if ("l" + str(line) == "l" + str(line_number)):
                        if feature_id_sorted.index(id) < feature_ind_of_selected_line:
                            feature_ind_of_selected_line = feature_ind_sorted[feature_id_sorted.index(id)]

        return feature_ind_of_selected_line

    def __getFeatureIndNodes__(self,feature_id_sorted,nodes,feature_ind_sorted):
        feature_ind_of_selected_node = len(feature_id_sorted)

        for node in nodes:
            for id in feature_id_sorted:
                if id[0] == "n":
                    node_number = int(''.join(list(filter(str.isdigit, id))))
                    if ("n" + str(node) == "n" + str(node_number)):
                        if feature_id_sorted.index(id) < feature_ind_of_selected_node:
                            feature_ind_of_selected_node = feature_ind_sorted[feature_id_sorted.index(id)]


        return feature_ind_of_selected_node

    def __getFeatureIndOfImportantConnectedLines__(self,node_ids,lines,feature_id_sorted,feature_ind_sorted):

        conn = {}
        for id in node_ids:
            node_number = int(''.join(list(filter(str.isdigit, id))))
            if node_number not in conn:
                conn[node_number] = {}
                conn[node_number]["line"] = self.__getConnectedLines__(node_number,lines)

        for node in conn:
            conn[node]["feature"] = self.__getFeatureIndLines__(feature_id_sorted,conn[node]["line"],feature_ind_sorted)


        return conn

    def __getFeatureIndOfImportantConnectedNodes__(self,line_ids, lines,feature_id_sorted,feature_ind_sorted):

        conn = {}
        for id in line_ids:
            line_number = int(''.join(list(filter(str.isdigit, id))))
            if line_number not in conn:
                conn[line_number] = {}
                conn[line_number]["node"] = self.__getConnectedNodes__(line_number, lines)

        for line in conn:
            conn[line]["feature"] = self.__getFeatureIndNodes__(feature_id_sorted, conn[line]["node"],feature_ind_sorted)


        return conn

    def __getSelectedFeatures__(self,feature_ind_sorted,feature_id_sorted,dict_of_node_with_selected_line_feature_ind,dict_of_line_with_selected_node_feature_ind,cons):
        selected_features = []

        ## TODO : improve the algorithm here!
        if cons:
            for i in feature_id_sorted:
                if i[0] == "n":
                    node_number = int(''.join(list(filter(str.isdigit, i))))
                    if feature_ind_sorted[feature_id_sorted.index(i)] not in selected_features:
                        selected_features.append(feature_ind_sorted[feature_id_sorted.index(i)])
                        for j in feature_id_sorted:
                            if j[0] == "n":
                                node_number2 = (int(''.join(list(filter(str.isdigit, j)))))
                                if (node_number2 == node_number) and (
                                        feature_ind_sorted[feature_id_sorted.index(j)] not in selected_features):
                                    selected_features.append(feature_ind_sorted[feature_id_sorted.index(j)])
                        if dict_of_node_with_selected_line_feature_ind[node_number]['feature'] not in selected_features:
                            selected_features.append(dict_of_node_with_selected_line_feature_ind[node_number]['feature'])
                # if the line has a better rank than a node
                else:
                    line_number = int(''.join(list(filter(str.isdigit, i))))
                    if feature_ind_sorted[feature_id_sorted.index(i)] not in selected_features:
                        selected_features.append(feature_ind_sorted[feature_id_sorted.index(i)])
                        if dict_of_line_with_selected_node_feature_ind[line_number]['feature'] not in selected_features:
                            selected_features.append(dict_of_line_with_selected_node_feature_ind[line_number]['feature'])
                            node_feature_id = feature_id_sorted[feature_ind_sorted.index(
                                dict_of_line_with_selected_node_feature_ind[line_number]['feature'])]
                            node_number3 = (int(''.join(list(filter(str.isdigit, node_feature_id)))))
                            for j in feature_id_sorted:
                                if j[0] == "n":
                                    node_number4 = (int(''.join(list(filter(str.isdigit, j)))))
                                    node_feature_ind = feature_ind_sorted[feature_id_sorted.index(j)]
                                    if (node_number4 == node_number3) and (node_feature_ind not in selected_features):
                                        selected_features.append(node_feature_ind)
                                ##if reactive power flow has been considered
                                if j[0] == "l":
                                    line_number2 = (int(''.join(list(filter(str.isdigit, j)))))
                                    line_feature_ind = feature_ind_sorted[feature_id_sorted.index(j)]
                                    if (line_number2 == line_number) and (line_feature_ind not in selected_features):
                                        selected_features.append(line_feature_ind)

            if len(selected_features) < len(feature_id_sorted):
                selected_features.extend(list(set(feature_ind_sorted)-set(selected_features)))
        else:
            selected_features = feature_ind_sorted
        return selected_features

    def getImportantFeatures(self,input_data,output_data,meter_placement_constraint=True):

        """ This function tries to find the most relevant features for a certain input-output data set
            For every grid the input data set is same but the output data is changed for every child grid"""

        '''
        lines = np.zeros([len(self.grid.line_dict), 4])

        k = 0
        for key in self.grid.line_dict:
            val = self.grid.line_dict[key]
            lines[k, :] = [key, val[0], val[1], val[2]]
            k = k + 1
        '''



        scores = mutual_info_classif(input_data, output_data)



        all_features_with_unsorted_scores = np.vstack((scores, self.grid.feature_inds))

        all_features_with_sorted_scores = all_features_with_unsorted_scores[:, (-all_features_with_unsorted_scores[0, :]).argsort()]

        feature_ind_sorted_unconstrained = list(np.ndarray.astype(all_features_with_sorted_scores[1,:],dtype='int'))

        feature_id_sorted_unconstrained = [self.grid.feature_ids[feature_ind] for feature_ind in feature_ind_sorted_unconstrained]

        dict_of_nodes_with_selected_line_feature_ind = self.__getFeatureIndOfImportantConnectedLines__(self.grid.node_feature_ids,self.grid.line_dict,feature_id_sorted_unconstrained,feature_ind_sorted_unconstrained)

        dict_of_lines_with_selected_node_feature_ind = self.__getFeatureIndOfImportantConnectedNodes__(self.grid.line_feature_ids,self.grid.line_dict, feature_id_sorted_unconstrained,feature_ind_sorted_unconstrained)

        feature_ind_sorted_constrained = self.__getSelectedFeatures__(feature_ind_sorted_unconstrained, feature_id_sorted_unconstrained,
                                                dict_of_nodes_with_selected_line_feature_ind,
                                                dict_of_lines_with_selected_node_feature_ind, meter_placement_constraint)


        feature_id_sorted_constrained = [feature_id_sorted_unconstrained[feature_ind_sorted_unconstrained.index(feature_ind)] for feature_ind in feature_ind_sorted_constrained]

        self.status = "Grid features sorted"
        print(self.status)

        if self.child_grid == None:
            self.grid.optimal_feature_inds_for_total_classification = feature_ind_sorted_constrained
            self.grid.optimal_feature_ids_for_total_classification = feature_id_sorted_constrained

            return self.grid
        else:
            self.child_grid.optimal_feature_inds_for_total_classification = feature_ind_sorted_constrained
            self.child_grid.optimal_feature_ids_for_total_classification = feature_id_sorted_constrained

            return self.child_grid



