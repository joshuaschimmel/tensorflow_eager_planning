import pendulum_scripts as pend
import csv

pend.create_training_data(iterations=10000)

# with open("pendulum_data.csv") as csvfile:
#     reader = csv.DictReader(csvfile)
#     print(reader.fieldnames)
#     print(reader.__next__())

#pend.example_run()
