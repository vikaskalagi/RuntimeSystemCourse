// agent/MethodAdvisor.java
package agent;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import net.bytebuddy.asm.Advice;

import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.util.*;

public class MethodAdvisor {

    private static final Gson gson = new GsonBuilder().setPrettyPrinting().create();

    @Advice.OnMethodEnter
    static long onEnter(@Advice.Origin String method,
                        @Advice.AllArguments Object[] args) {

        // store start-nanos in local variable
        long start = System.nanoTime();

        // store metadata snapshot for exit
        MethodContext.set(method, args, start);

        return start;
    }

    @Advice.OnMethodExit(onThrowable = Throwable.class)
    static void onExit(@Advice.Enter long start,
                       @Advice.Return Object returned) {

        MethodContext ctx = MethodContext.get();
        if (ctx == null) return;

        long durationNs = System.nanoTime() - ctx.startTime;

        Map<String, Object> event = new LinkedHashMap<>();
        event.put("timestamp", System.currentTimeMillis());
        event.put("language", "Java");

        event.put("execution", Map.of(
                "method", ctx.method,
                "thread", Thread.currentThread().getName()
        ));

        Runtime r = Runtime.getRuntime();

        event.put("memory", Map.of(
                "heap_used", r.totalMemory() - r.freeMemory(),
                "heap_committed", r.totalMemory(),
                "heap_max", r.maxMemory()
        ));

        event.put("type", Map.of(
                "return_type", returned == null ? "void" : returned.getClass().getName(),
                "is_class", returned instanceof Class
        ));

        // argument types
        List<String> argTypes = new ArrayList<>();
        for (Object arg : ctx.args) {
            argTypes.add(arg == null ? "null" : arg.getClass().getName());
        }

        event.put("dynamic", Map.of(
                "arguments", argTypes
        ));

        event.put("performance", Map.of(
                "exec_time_ns", durationNs,
                "exec_time_ms", durationNs / 1_000_000.0
        ));

        // GC info
        List<Map<String, Object>> gcEvents = new ArrayList<>();
        for (GarbageCollectorMXBean gc : ManagementFactory.getGarbageCollectorMXBeans()) {
            gcEvents.add(Map.of(
                    "name", gc.getName(),
                    "count", gc.getCollectionCount(),
                    "time_ms", gc.getCollectionTime()
            ));
        }
        event.put("gc", gcEvents);

        System.out.println(gson.toJson(event));
    }
}
