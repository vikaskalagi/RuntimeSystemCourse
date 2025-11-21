package agent;

public class MethodContext {

    public final String method;
    public final Object[] args;
    public final long startTime;

    private static final ThreadLocal<MethodContext> local = new ThreadLocal<>();

    public MethodContext(String method, Object[] args, long startTime) {
        this.method = method;
        this.args = args;
        this.startTime = startTime;
    }

    public static void set(String method, Object[] args, long start) {
        local.set(new MethodContext(method, args, start));
    }

    public static MethodContext get() {
        return local.get();
    }
}
