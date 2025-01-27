
class NestedDictToObject:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                # Recursively convert nested dictionaries into objects
                value = NestedDictToObject(value)
            setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, key):
        # Allow dictionary-style access
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow setting attributes dictionary-style
        setattr(self, key, value)

    def __contains__(self, key):
        # Check if a key exists
        return hasattr(self, key)

    def to_dict(self):
        # Convert back to a dictionary
        result = {}
        for key in self.__dict__:
            value = getattr(self, key)
            if isinstance(value, NestedDictToObject):
                value = value.to_dict()
            result[key] = value
        return result


# Example usage
def test_object():
    nested_dict = {
        'key1': {
            'next_key1': {
                'final_key': 'value1'
            },
            'next_key2': 'value2'
        },
        'key2': 'value3'
    }

    # Convert the nested dictionary to an object
    obj = NestedDictToObject(nested_dict)

    # Access values using attribute-style access
    print(obj.key1.next_key1.final_key)  # Outputs: value1
    print(obj.key1.next_key2)            # Outputs: value2
    print(obj.key2)                      # Outputs: value3

    # Convert back to a dictionary
    print(obj.to_dict())


if __name__ == "__main__":
    test_object()
