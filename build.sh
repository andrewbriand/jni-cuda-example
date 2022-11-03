echo "JAVA_HOME: $JAVA_HOME"
javac -h . VectorAdd.java
nvcc -Xcompiler -fPIC -I"$JAVA_HOME/include" -I"$JAVA_HOME/include/linux" -shared -o libvectoradd.so VectorAdd.cu
scalac VectorAddScala.scala
