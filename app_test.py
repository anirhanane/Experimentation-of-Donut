import utils as u
def main():
    print("---Load Data---")
    dataset=u.load_database_sroie()
    print("---Print a Sample---")
    u.view_dataset(dataset)
    print("---Process Datset for Donut---")
    proc_dataset=u.preprocess_documents_for_donut(dataset)
    print("---Tokenize Data---")
    processor=u.load_DonutProcessor()
    processed_dataset=processed_dataset(proc_dataset)
    print("---Fin---")


if __name__=="__main__":
    main()