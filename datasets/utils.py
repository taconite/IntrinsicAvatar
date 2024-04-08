class KeyIndex:
    def __init__(self, values, key=None):
        # If 'values' is a list of dictionaries, store it directly. Otherwise, create a list of
        # dictionaries from the values and key.
        if all(isinstance(v, dict) for v in values):
            self.data = values
        else:
            self.data = [{key: value} for value in values]

    def __mul__(self, other):
        if not isinstance(other, KeyIndex):
            raise ValueError("Operand must be an instance of KeyIndex")

        result = []
        for dict1 in self.data:
            for dict2 in other.data:
                merged_dict = {**dict1, **dict2}  # Merge dictionaries
                result.append(merged_dict)
        return KeyIndex(result)

    def __add__(self, other):
        if not isinstance(other, KeyIndex) or len(self.data) != len(other.data):
            raise ValueError(
                "Operands must be instances of KeyIndex with equal length data"
            )

        result = []
        for dict1, dict2 in zip(self.data, other.data):
            merged_dict = {**dict1, **dict2}  # Merge dictionaries
            result.append(merged_dict)
        return KeyIndex(result)

    def to_list(self):
        return self.data


