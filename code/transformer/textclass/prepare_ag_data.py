import csv

# transforms the AG News dataset into the format expected in our script
# csv to tsv, removing the unnecessary quotes
# mapping the labels to the corresponding labels in the Hausa and Yoruba datasets

# labels in AG
# Sports', 'World', 'Business', 'Sci/Tech
# labels for yoruba
# "nigeria", "africa", "world", "entertainment", "health", "sport", "politics"
label_map_yoruba = {"Sports":"sport", "World":"world","Business":"business","Sci/Tech":"sci/tech"}
label_map_hausa = None

def prepare_dataset(in_path, out_path, label_map):
    with open(in_path, "r") as in_file:
        with open(out_path, "w") as out_file:
            in_reader = csv.reader(in_file, delimiter=',', quotechar='"')
            out_file.write("newstitle\tlabel\n")
            for row in in_reader:
                assert len(row) == 2
                assert not "\t" in row[0]
                label = row[1]
                if not label_map is None:
                    label = label_map[label]
                out_file.write(f"{row[0]}\t{label}\n")

if __name__ == "__main__":
    prepare_dataset("../../data/new/ag-news/AG/orig_download/eval/AG__TEST.csv",
                    "../../data/new/ag-news/AG/ag-for-yoruba/test.csv",
                    label_map_yoruba)
    prepare_dataset("../../data/new/ag-news/AG/orig_download/training/AG__FULL.csv",
                    "../../data/new/ag-news/AG/ag-for-yoruba/train.csv",
                    label_map_yoruba)

    prepare_dataset("../../data/new/ag-news/AG/orig_download/eval/AG__TEST.csv",
                    "../../data/new/ag-news/AG/ag-for-hausa/test.csv",
                    label_map_hausa)
    prepare_dataset("../../data/new/ag-news/AG/orig_download/training/AG__FULL.csv",
                    "../../data/new/ag-news/AG/ag-for-hausa/train.csv",
                    label_map_hausa)

