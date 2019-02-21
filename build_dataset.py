import csv
import os
import glob

def main(arg):
    d = {}
    with open('./sample_lists/sample_list.csv', 'w') as s:
        with open('./sample_lists/labels.csv', 'r') as f:
            reader = csv.reader(f)
            for k, v in reader:
                file_list = glob.glob('./101_ObjectCategories/'+str(v)+'/*.jpg')
                for file_name in file_list:
                    print(str(file_name)+','+str(k)+'\n')
                    s.write(str(file_name)+','+str(k)+'\n')
                #d[str(v)] = k
            
if __name__ == "__main__":
    main(None)
