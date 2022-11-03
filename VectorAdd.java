import java.io.*;

public class VectorAdd {
  static {
    System.loadLibrary("vectoradd");
  }

  static native int[] add(int[] a, int[] b);
}
