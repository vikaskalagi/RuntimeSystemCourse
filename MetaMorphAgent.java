// File: MetaMorphAgent.java
import java.lang.instrument.*;
import java.lang.reflect.*;
import java.security.ProtectionDomain;

public class MetaMorphAgent {
    public static void premain(String args, Instrumentation inst) {
        System.out.println("MetaMorph Java Agent active.");
        inst.addTransformer(new ClassFileTransformer() {
            public byte[] transform(Module module, ClassLoader loader, String className,
                                    Class<?> classBeingRedefined, ProtectionDomain domain,
                                    byte[] classfileBuffer) {
                // Here you could inject bytecode to log runtime info
                System.out.println("Loaded class: " + className);
                return classfileBuffer;
            }
        });
    }

    public static void analyzeObject(Object obj) {
        Class<?> clazz = obj.getClass();
        System.out.println("Class: " + clazz.getName());
        for (Field field : clazz.getDeclaredFields()) {
            field.setAccessible(true);
            try {
                System.out.println("  Field: " + field.getName() + " = " + field.get(obj));
            } catch (Exception e) {}
        }
        for (Method m : clazz.getDeclaredMethods()) {
            System.out.println("  Method: " + m.getName());
        }
    }
}

// javac MetaMorphAgent.java
// jar cmf manifest.txt metamorph.jar MetaMorphAgent.class
// java -javaagent:metamorph.jar -cp . YourProgram

