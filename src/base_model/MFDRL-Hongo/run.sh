#!bin/bash

# Wait until hongo is built
echo ${HONGO_BUILD_DIR} || exit 1
while [ ! -f ${HONGO_BUILD_DIR}/_hongo.so ]; do
  sleep 1
done

python3 code/main.py ${MFD_RL_HONGO_CONFIG} ${INITIALIZE}
