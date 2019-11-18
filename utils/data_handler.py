def data_train_splitter():

    # TODO Find a better way to skip the header on the data_train.csv file
    urm_path = "../data/data_train_no_header.csv"
    urm_file = open(urm_path, 'r')

    def row_split(row_string):
        split = row_string.split(",")
        split[2] = split[2].replace("\n", "")

        split[0] = int(split[0])
        split[1] = int(split[1])
        split[2] = float(split[2])

        result = tuple(split)

        return result

    urm_file.seek(0)
    urm_tuples = []

    for line in urm_file:
        urm_tuples.append(row_split(line))

    return urm_tuples
