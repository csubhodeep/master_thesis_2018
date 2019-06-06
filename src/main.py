import numpy as np
import os
from src.data_acq import DataCollector
from src.feature_selection import FeatureSorter
from src.training import ModelTrainer
from src.create_partition import GridSplitter
from sklearn.model_selection import train_test_split
import time
from joblib import load,dump
from src.create_grid_object import Grid
import warnings
from copy import deepcopy


class PipelineExecutor:

    def __init__(self,cwd_path=os.getcwd()):
        self.status = "Starting Pipeline"
        print(self.status)
        self.cwd_path = cwd_path
        self.grid = Grid()
        self.dict_of_all_grids = {}

    def get_filtered_data(self,input_data,features):
        rows = input_data.shape[0]
        cols = len(features)

        input_data_sel = np.zeros([rows,cols])

        buff =np.zeros([rows,1])

        flg = 0
        for feature in features:
            if flg == 0:
                input_data_sel = input_data[:,int(feature)]
                flg = 1
                input_data_sel = input_data_sel.reshape(-1,1)
            else:
                buff[:,0] = input_data[:,int(feature)]
                input_data_sel = np.hstack((input_data_sel,buff))

        return input_data_sel

    def check_all_acc(self, split_method, list_of_grids = None, threshold = 0.93, min_n_classes = 7):

        list_of_weak_grids = []
        list_of_good_grids = []
        acc = []
        flg = False
        if split_method == "sequential" or split_method == "exponential":
            last_key = max(list(self.dict_of_all_grids))
            last_dict = dict(self.dict_of_all_grids[last_key])
            for grid_id,grid in last_dict.items():
                acc.append(grid.total_classification_accuracy)
                if grid.total_classification_accuracy < threshold:
                    list_of_weak_grids.append(grid)
            if len(list_of_weak_grids) != 0:
                flg = True

        else:
            for grid in list_of_grids:
                acc.append(grid.total_classification_accuracy)
                if grid.total_classification_accuracy < threshold and len(grid.line_list)>min_n_classes-1:
                    list_of_weak_grids.append(grid)
            if len(list_of_weak_grids)<=len(list_of_grids) and len(list_of_weak_grids):
                flg = True

        for grid in list_of_grids:
            if grid not in list_of_weak_grids:
                list_of_good_grids.append(grid)
        print(acc)


        return flg, list_of_weak_grids, list_of_good_grids

    def get_transformed_output_data(self,old_input_data,old_output_data,grid=None):
        transformed_output = np.array(old_output_data)

        if grid==None:
            for i in range(old_output_data.shape[0]):
                if old_output_data[i,0] in self.grid.total_clf_labels_dict:
                    transformed_output[i, 0] = self.grid.total_clf_labels_dict[old_output_data[i, 0]]
                else:
                    transformed_output[i,0] = "0_0"
        else:
            for i in range(old_output_data.shape[0]):
                if old_output_data[i, 0] in grid.total_clf_labels_dict:
                    transformed_output[i, 0] = grid.total_clf_labels_dict[old_output_data[i, 0]]
                else:
                    transformed_output[i, 0] = "0_0"


        number_of_samples_for_each_label_for_some_fault = []
        if grid==None:
            for label in self.grid.total_clf_labels_dict:
                if label != "0_0":
                    number_of_samples_for_each_label_for_some_fault.append(len(np.where(transformed_output == label)[0]))
        else:
            for label in grid.total_clf_labels_dict:
                if label != "0_0":
                    number_of_samples_for_each_label_for_some_fault.append(len(np.where(transformed_output == label)[0]))

        avg_number_of_samples_for_any_fault = int(np.mean(number_of_samples_for_each_label_for_some_fault))



        indices_for_no_fault = np.where(transformed_output=="0_0")[0]

        new_indices_for_no_fault = np.random.permutation(indices_for_no_fault)[:avg_number_of_samples_for_any_fault]

        indices_for_some_fault = np.where(transformed_output!="0_0")[0]

        new_indices = np.hstack((new_indices_for_no_fault,indices_for_some_fault))


        modified_transformed_output = transformed_output[new_indices,0]

        modified_transformed_output = modified_transformed_output.reshape(-1,1)


        transformed_input = old_input_data[new_indices,:]

        return modified_transformed_output, transformed_input

    def split_into_sub_zones(self,it,number_of_parts,split_method,parent_grid=None):

        self.dict_of_all_grids[it] = {}
        flg=False
        if split_method == "exponential":
            for grid_id in list(self.dict_of_all_grids[it]):
                if it == 0:
                    list_of_child_grids, self.dict_of_all_grids[it][grid_id], flg = GridSplitter(self.dict_of_all_grids[it][grid_id]).make_sub_zones(number_of_parts,split_method)
                else:
                    list_of_child_grids, self.dict_of_all_grids[it][grid_id], flg = GridSplitter(self.dict_of_all_grids[it][grid_id],self.dict_of_all_grids[0]["r"]).make_sub_zones(number_of_parts,split_method)

                for child_grid in list_of_child_grids:
                    self.dict_of_all_grids[it + 1][child_grid.grid_id] = child_grid
        elif split_method == "sequential":
            list_of_child_grids, self.dict_of_all_grids[0]["r"], flg = GridSplitter(
                self.dict_of_all_grids[0]["r"]).make_sub_zones(number_of_parts, split_method)

            for child_grid in list_of_child_grids:
                self.dict_of_all_grids[it + 1][child_grid.grid_id] = child_grid
        else:
            list_of_child_grids, parent_grid, flg = GridSplitter(
                parent_grid).make_sub_zones(number_of_parts, split_method)


        return flg, parent_grid, list_of_child_grids

    def train_for_total_clfn(self, input_data,output_data, it,grid=None, number_of_samples=-1):

        transformed_output_data, transformed_input_data = self.get_transformed_output_data(input_data, output_data,grid)


        train_input, test_input, train_output, test_output = train_test_split(transformed_input_data, transformed_output_data)
        #del input_data, output_data, transformed_output_data
        ### take first 4 features (1 PMU installed in 1 location/node), train a model and then obtain the accuracy

        if grid==None:

            self.dict_of_all_grids[it][self.grid.grid_id] = FeatureSorter(self.grid,grid).getImportantFeatures(train_input[0:number_of_samples, :],train_output[0:number_of_samples,:])
            important_features = list(self.dict_of_all_grids[it][self.grid.grid_id].optimal_feature_inds_for_total_classification[:4])
            train_input_data_selected = self.get_filtered_data(train_input, important_features)
            test_input_data_selected = self.get_filtered_data(test_input, important_features)
            self.dict_of_all_grids[it][self.grid.grid_id] = ModelTrainer(self.dict_of_all_grids[it][self.grid.grid_id]).train(train_input_data_selected,
                                                                                         test_input_data_selected,
                                                                                         train_output,
                                                                                         test_output,training_method='rf')
            return self.dict_of_all_grids[it][self.grid.grid_id]
        else:
            self.dict_of_all_grids[it][grid.grid_id] = FeatureSorter(self.grid,grid).getImportantFeatures(train_input[0:number_of_samples, :],train_output[0:number_of_samples,:])
            important_features = list(self.dict_of_all_grids[it][grid.grid_id].optimal_feature_inds_for_total_classification[:4])
            train_input_data_selected = self.get_filtered_data(train_input, important_features)
            test_input_data_selected = self.get_filtered_data(test_input, important_features)
            self.dict_of_all_grids[it][grid.grid_id] = ModelTrainer(self.dict_of_all_grids[it][grid.grid_id]).train(train_input_data_selected,
                                                                               test_input_data_selected, train_output,
                                                                               test_output,training_method='rf')



            return self.dict_of_all_grids[it][grid.grid_id]

    def train_for_zone_clfn(self, old_input_data, old_output_data, last_iteration):

        ## build a combined model here by first defining a predictor class

        # transform output data here

        output_data_mapper = {}
        output_data_mapper['0_0']='0_0'
        all_optimal_feature_inds = []
        for grid_id, grid in self.dict_of_all_grids[last_iteration].items():
            all_optimal_feature_inds.extend(grid.optimal_feature_inds_for_total_classification[:4])
            for k,v in grid.total_clf_labels_dict.items():
                if v != '0_0':
                    output_data_mapper[v] = grid_id

        # filter out measurements from input data here

        new_output_data = np.array(old_output_data)
        for i in range(old_output_data.shape[0]):
            new_output_data[i,0] = output_data_mapper[old_output_data[i,0]]

        new_input_data = self.get_filtered_data(old_input_data, all_optimal_feature_inds)

        train_input_data_selected, test_input_data_selected, train_output, test_output = train_test_split(new_input_data,new_output_data)


        self.dict_of_all_grids[0]["r"] = ModelTrainer(self.dict_of_all_grids[0]["r"],train_total=False).train(
            train_input_data_selected,
            test_input_data_selected, train_output,
            test_output, training_method='rf')

        return None



    def get_accuracy(self,input_data,output_data,split_method,number_of_samples=-1,step=1):

        iteration = 0


        self.dict_of_all_grids[iteration] = {self.grid.grid_id: self.grid}
        unused = self.train_for_total_clfn(input_data,output_data, iteration,number_of_samples=number_of_samples)



        flg, unused, unused = self.check_all_acc(split_method,list_of_grids=[self.dict_of_all_grids[0]["r"]])

        if split_method == "exponential":
            while flg:
                for _,__ in self.dict_of_all_grids[iteration].items():
                    flg1, unused1, unused2 = self.split_into_sub_zones(iteration,2,split_method)
                    for grid_id2, grid2 in self.dict_of_all_grids[iteration+1].items():
                        unused = self.train_for_total_clfn(input_data,output_data,iteration+1,grid2,number_of_samples=number_of_samples)
                    if flg1:
                        flg, unused, unused = self.check_all_acc(split_method)
                    else:
                        flg = flg1
                    if not(flg):
                        break
                    else:
                        iteration=iteration+step
        elif split_method == "sequential":
            while flg:

                if iteration == 0:
                    n_parts = 2
                else:
                    n_parts = iteration * step + 2
                flg1, unused, unused = self.split_into_sub_zones(iteration,n_parts,split_method)
                for grid_id2, grid2 in self.dict_of_all_grids[iteration+1].items():
                    unused = self.train_for_total_clfn(input_data,output_data,iteration+1,grid2,number_of_samples=number_of_samples)

                if flg1:
                    flg, unused, unused = self.check_all_acc(split_method)
                else:
                    flg = flg1
                iteration=iteration+1
        else:
            list_of_good_grids = []
            iteration = iteration + 1
            while flg:

                n_parts = iteration + step

                err_flg, parent_grid, list_of_child_grids = self.split_into_sub_zones(iteration, n_parts, split_method,self.grid)
                list_of_trained_child_grids = []
                for grid in list_of_child_grids:
                    list_of_trained_child_grids.append(self.train_for_total_clfn(input_data, output_data, iteration, grid, number_of_samples=number_of_samples))
                split_flg, list_of_weak_grids, list_of_good_grids = self.check_all_acc(split_method,list_of_trained_child_grids)
                if err_flg:
                    flg = split_flg
                if flg and len(list_of_weak_grids) == len(list_of_trained_child_grids):
                    iteration = iteration + 1
                else:
                    """ do splitting in the weak grids as long as each weak grid gets split into strong grids, 
                    then move on to the next weak grid """
                    for weak_grid in list_of_weak_grids:
                        flg3 = 1
                        sub_iteration = 1
                        sub_step = 1
                        new_parent_grid = deepcopy(weak_grid)
                        while flg3:
                            sub_n_parts = sub_iteration + sub_step
                            err_flg2, new_parent_grid, list_of_child_grids2 = self.split_into_sub_zones(iteration, sub_n_parts,
                                                                                               split_method,new_parent_grid)
                            list_of_trained_child_grids2 = []
                            for grid in list_of_child_grids2:
                                list_of_trained_child_grids2.append(
                                    self.train_for_total_clfn(input_data, output_data, iteration, grid,
                                                              number_of_samples=number_of_samples))
                            split_flg2, list_of_weak_grids2, list_of_good_grids2 = self.check_all_acc(split_method, list_of_trained_child_grids2)

                            if err_flg2:
                                flg3 = split_flg2
                            if flg3:
                                if len(list_of_weak_grids2) == len(list_of_trained_child_grids2):
                                    sub_iteration = sub_iteration + 1
                                else:
                                    list_of_weak_grids.extend(list_of_weak_grids2)
                                    sub_iteration = sub_iteration + 1
                                    list_of_good_grids.extend(list_of_good_grids2)
                                    flg3 = False
                            else:
                                list_of_good_grids.extend(list_of_good_grids2)

                        del new_parent_grid
                    flg = False

            for child_grid in list_of_good_grids:
                self.dict_of_all_grids[iteration][child_grid.grid_id] = child_grid

        return None

    def populate_grid_object(self,all_data):

        ## start populating the parent grid object from here

        self.status = "Populating grid object"
        print(self.status)

        self.grid.parent_grid_id = "r"
        self.grid.grid_id = self.grid.parent_grid_id
        self.grid.node_list = list(all_data['node_list'])
        self.grid.line_dict = dict(all_data['line_dict'])
        self.grid.node_feature_ids = list(all_data['node_ids'])
        self.grid.line_feature_ids = list(all_data['line_ids'])

        node_feature_ind = [i for i in range(len(all_data['node_ids']))]
        line_feature_ind = [len(all_data['node_ids']) + i for i in range(len(all_data['line_ids']))]

        self.grid.feature_ids = list(all_data['node_ids'])
        self.grid.feature_ids.extend(all_data['line_ids'])
        self.grid.feature_inds = list(node_feature_ind)
        self.grid.feature_inds.extend(line_feature_ind)

        for line in self.grid.line_dict:
            val = self.grid.line_dict[line]
            #if val[2] != 0 and [val[0],val[1]] not in self.grid.line_list:
            if [val[0], val[1]] not in self.grid.line_list:
                self.grid.line_list.append([val[0],val[1]])

        if len(self.grid.line_list) == len(self.grid.node_list) - 1:
            self.grid.grid_type = "radial"
        else:
            self.grid.grid_type = "non-radial"

        for node in self.grid.node_list:
            self.grid.node_dict[node] = []
            for line in self.grid.line_dict:
                fb = self.grid.line_dict[line][0]
                tb = self.grid.line_dict[line][1]
                if (fb == node or tb == node) and (line not in self.grid.node_dict[node]):
                    self.grid.node_dict[node].append(line)

        for i in range(all_data['output_data'].shape[0]):
            if all_data['output_data'][i,0] not in self.grid.total_clf_labels_dict:
                self.grid.total_clf_labels_dict[all_data['output_data'][i,0]] = all_data['output_data'][i,0]



        self.status = "Grid object populated"
        print(self.status)

        return None

    def start(self,extract_data=False,split_method="exponential",write_to_file=False):

        t_start = time.time()
        if extract_data:
            DataCollector(self.cwd_path).get_all_data()



        self.status = "Loading grid data from disk"
        print(self.status)
        all_data = load(self.cwd_path+"/scenario_data/all_data_123_a.pkl")

        self.populate_grid_object(all_data)


        input_data, validation_input, output_data, validation_output = train_test_split(all_data['input_data'], all_data['output_data'])

        dump({'validation_input':validation_input, 'validation_output':validation_output},os.getcwd()+'/validation_data.pkl')

        self.get_accuracy(input_data,output_data,split_method,number_of_samples=2000,step=11)



        last_key = max(list(self.dict_of_all_grids))

        self.train_for_zone_clfn(input_data, output_data, last_key)

        #self.dict_of_all_grids[0]["r"].see_values(save_to_file=write_to_file)

        dict_of_classifiers = {}
        self.dict_of_all_grids[last_key]["r"] = self.dict_of_all_grids[0]["r"]
        dump(self.dict_of_all_grids[last_key], 'dict_of_grids.pkl')
        dict_of_classifiers["r"] = self.dict_of_all_grids[0]["r"].zone_classifier_model
        for grid_id,grid in self.dict_of_all_grids[last_key].items():
            dict_of_classifiers[grid_id] = grid.total_classifier_model
            grid.see_values(save_to_file=write_to_file)
        dump(dict_of_classifiers,'dict_of_classifiers.pkl')




        self.status = "Pipeline finished"
        print(self.status)
        print("Total execution time: "+str(round(time.time() - t_start))+"s")
        
        return None


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        PipelineExecutor().start(split_method="advanced",write_to_file=True)


