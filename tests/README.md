# Test suite

SchNetPack has unit tests that you can run to make sure that everything is working properly.

## Test dependencies
To install the additional test dependencies, install with:
```
pip install schnetpack[test]
```

Or, if you installed from source and assuming you are in your local copy of the SchNetPack repository,
```
pip install .[test]
```

## Run tests
In order to run the tests, run the following command from the root of the repository:
```
pytest tests
```
