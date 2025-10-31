class Dog:
    species = "Canis familiaris"

    def __init__(self, name, age):
        self.name = name  # Instance attribute
        self.age = age    # Instance attribute

    def bark(self):
        return f"{self.name} says woof!"

    def get_info(self):
        return f"{self.name} is a {self.species} and is {self.age} years old."

# Create instances (objects) of the Dog class
dog1 = Dog("Buddy", 3)
dog2 = Dog("Lucy", 5)

# Access attributes and call methods of the objects
print(dog1.name)
print(dog2.age)
print(dog1.bark())
print(dog2.get_info())