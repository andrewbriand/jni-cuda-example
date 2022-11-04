This is a simple example of launching a CUDA kernel in Scala using the [Java Native Interface](https://docs.oracle.com/en/java/javase/17/docs/specs/jni/intro.html) (JNI). 

### Building and running

First, make sure that `JAVA_HOME` is set appropriately and that `javac` and `scalac` are available on your `PATH`. To do this, you can run the following commands:

```
export JAVA_HOME=/PATH/TO/YOUR/JDK/
export PATH=/PATH/TO/YOUR/JDK/bin/:$PATH
export PATH=/PATH/TO/YOUR/SCALA/bin/:$PATH
```

Also, please make sure you have the CUDA toolkit installed and that `nvcc` is available on your `PATH`. Then, run:

```
./build.sh
```

This will create a JNI header file for the example Java method, compile its implementation with nvcc, and compile the scala code in `VectorAddScala.scala`. Look at the comments in `build.sh` for more details. To run:

```
./run.sh
```

which simply executes the resulting program with `scala`. The program creates two integer arrays, adds them together elementwise with a CUDA kernel, and prints out the beginning and end of the resulting array.
