package agent;

import net.bytebuddy.agent.builder.AgentBuilder;
import net.bytebuddy.asm.Advice;

import java.lang.instrument.Instrumentation;

public class RuntimeProbeAgent {

    public static void premain(String agentArgs, Instrumentation inst) {
        System.out.println("[MetaMorph] Java Runtime Probe Activated");

        new AgentBuilder.Default()
            .type((typeMatcher) -> true)   // instrument all classes (filter later)
            .transform((builder, typeDescription, classLoader, module) ->
                builder.visit(Advice.to(MethodAdvisor.class)
                .on((m) -> !m.isConstructor()
                        && !m.isAbstract()
                        && !m.isNative())))
            .installOn(inst);
    }
}
