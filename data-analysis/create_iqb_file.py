
from iqb_io import *
from iqb_saa_benchmarks import *


if __name__=="__main__":
    with open("table_with_results_1.txt") as file:
        bla = file.readlines()
        print(bla)
        b = bla.split("----------------------")
        print(b)
        final = {}
        classifier_results = []
        clas_name = ""
        for n in bla:
            if n.endswith("_bow") or n.endswith("_sim"):
                if len(classifier_results) != 0:
                    final[clas_name] = classifier_results
                    classifier_results = []
                    clas_name = n
                else:
                    clas_name = n
            else:
                classifier_results.append(n)
        for i in final.keys():
            print("---------------PRINTING :{}".format(i))
            for x in final[i]:
                print(x)
            print("-------------------------------------------")



