cython observation_utils.pyx  -3 -Wextra -D
cmd="gcc-7 observation_utils.c `python3-config --cflags --ldflags --includes --libs` -fno-lto -pthread -fPIC -fwrapv -pipe -march=native -mtune=native -Ofast -msse2 -msse4.2 -shared -o observation_utils.so"
echo "Executing $cmd"
$cmd
echo "Testing compilation.."
python3 -c "import observation_utils"
