import pandas as pd

class EllipticService:

    def get_label_by_transaction(self, transaction_id):
        classes_df = pd.read_csv("data/elliptic/elliptic_txs_classes.csv")

        transaction_class = classes_df[classes_df["txId"] == transaction_id]

        if transaction_class is None:
            return False
        else:
            return 'Illicit' if transaction_class.values[0][1] == '2' else 'Licit'
