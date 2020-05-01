
from iqb_io import *
from iqb_saa_benchmarks import *


if __name__=="__main__":
    iqb_data = read_iqb_data("iqb-tba-answers-doublequotes.tsv")
    var_data = pd.read_csv("varinfo.tsv", sep='\t')

    a = iqb_data["value.raw"]
    print(a.iloc[5239])
    print(a.iloc[38455])
    print(a.iloc[7046])
    print(a.iloc[50822])
    print(a.iloc[50823])
    print(a.iloc[50824])

    write_to_csv("iqb-tba-answers-with-meta-doublequotes.tsv")
