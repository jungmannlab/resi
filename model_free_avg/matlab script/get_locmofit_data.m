%path = "W:\users\reinhardt\RESI\locmofit-avg\RESI_cluster_centers_analysis\";
%data = load(path + "analysis250122_RESI_centers_avg");
%n_iter = 13;

%path = "W:\users\reinhardt\RESI\locmofit-avg\Eduard_analysis\";
%data = load(path + "DNA-PAINT_Eduard_filter2_avg");
%n_iter = 10;

path = "W:\users\reinhardt\RESI\locmofit-avg\Eduard_analysis\";
data = load(path + "DNA-PAINT_Eduard_filter2_avg");
n_iter = 13;


% data seems to have 13 subsets, let's explore them

data_avg = data.finalAvg;


for i = 1:n_iter
    
    locs = [data_avg{i}.xnm data_avg{i}.ynm data_avg{i}.znm];
    
    other_data = [data_avg{i}.locprecnm data_avg{i}.locprecznm data_avg{i}.channel data_avg{i}.layer];
    
    all = [data_avg{i}.xnm data_avg{i}.ynm data_avg{i}.znm data_avg{i}.locprecnm data_avg{i}.locprecznm data_avg{i}.channel data_avg{i}.layer];
    
    csvwrite(path + "avg_locs_" + string(i) + ".csv", locs)
    csvwrite(path + "avg_locs_other_data" + string(i) + ".csv", other_data)
    csvwrite(path + "avg_locs_all_" + string(i) + ".csv", all)
    
end


data.indBest
