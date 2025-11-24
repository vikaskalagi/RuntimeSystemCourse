public class Matrix {

    public static void main(String[] args) throws Exception {

        int[][] mat = new int[][] {
            {1, 2},
            {3, 4}
        };

        // Call the correct method
        String info = Probe.collectInfo(mat);

        System.out.println("[JAVA PROBE OUTPUT]");
        System.out.println(info);
    }
}
