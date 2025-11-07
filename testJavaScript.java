
public class Dog {
    static String species = "Canis familiaris";

    String name;
    int age;

    public Dog(String name, int age) {
        this.name = name;
        this.age = age;
    }

    public String bark() {
        return name + " says woof!";
    }

    public String getInfo() {
        return name + " is a " + species + " and is " + age + " years old.";
    }

    public static void main(String[] args) {
        
        Dog dog1 = new Dog("Buddy", 3);
        Dog dog2 = new Dog("Lucy", 5);

        System.out.println(dog1.name);
        System.out.println(dog2.age);
        System.out.println(dog1.bark());
        System.out.println(dog2.getInfo());
    }
}
