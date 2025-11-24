public class Probe {

    public static String collectInfo(Object obj) {
        Class<?> cls = obj.getClass();

        // Manual JSON assembly
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        sb.append("\"class_name\":\"").append(cls.getName()).append("\",");

        // Fields
        sb.append("\"fields\":[");
        var fields = cls.getDeclaredFields();
        for (int i = 0; i < fields.length; i++) {
            sb.append("\"").append(fields[i].getName()).append("\"");
            if (i < fields.length - 1) sb.append(",");
        }
        sb.append("],");

        // Methods
        sb.append("\"methods\":[");
        var methods = cls.getDeclaredMethods();
        for (int i = 0; i < methods.length; i++) {
            sb.append("\"").append(methods[i].getName()).append("\"");
            if (i < methods.length - 1) sb.append(",");
        }
        sb.append("]");

        sb.append("}");
        return sb.toString();
    }
}
