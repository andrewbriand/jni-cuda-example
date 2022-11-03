import java.io.*;

public class VectorAdd {
  static {
    System.loadLibrary("vectoradd");
  }

  private static native int[] add(int[] a, int[] b);

  public static void main(String[] args) {
    int len = 10;

    int[] a = new int[len];
    int[] b = new int[len];

    for (int i = 0; i < len; i++) {
      a[i] = i;
      b[i] = i;
    }

    int[] c = add(a, b);

    for (int i = 0; i < len; i++) {
      System.out.println(c[i]);
    }
  }
}
