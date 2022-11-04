echo "JAVA_HOME: $JAVA_HOME"

# Generate a C/C++ header file (VectorAdd.h) for the native method declared in VectorAdd.java
javac -h . VectorAdd.java

# Compile the CUDA code that will implement the java native method into a shared library
nvcc -Xcompiler -fPIC -I"$JAVA_HOME/include" -I"$JAVA_HOME/include/linux" -shared -o libvectoradd.so VectorAdd.cu

# Compile the whole scala program
scalac VectorAddScala.scala
