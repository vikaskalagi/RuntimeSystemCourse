#include <iostream>
#include <string>
using namespace std;

class Dog {
public:
    static string species;  

    Dog(string name, int age) {
        this->name = name;
        this->age = age;
    }

    string bark() {
        return name + " says woof!";
    }

    string get_info() {
        return name + " is a " + species + " and is " + to_string(age) + " years old.";
    }

    string name;
    int age;
};

string Dog::species = "Canis familiaris";

int main() {

    Dog dog1("Buddy", 3);
    Dog dog2("Lucy", 5);
    
    cout << dog1.name << endl;
    cout << dog2.age << endl;
    cout << dog1.bark() << endl;
    cout << dog2.get_info() << endl;

    return 0;
}
