# ram_project

# LSDIR links:

https://data.vision.ee.ethz.ch/yawli/shard-00.tar.gz
https://data.vision.ee.ethz.ch/yawli/shard-01.tar.gz
https://data.vision.ee.ethz.ch/yawli/shard-02.tar.gz
https://data.vision.ee.ethz.ch/yawli/shard-03.tar.gz
https://data.vision.ee.ethz.ch/yawli/shard-04.tar.gz
https://data.vision.ee.ethz.ch/yawli/shard-05.tar.gz
https://data.vision.ee.ethz.ch/yawli/shard-06.tar.gz
https://data.vision.ee.ethz.ch/yawli/shard-07.tar.gz
https://data.vision.ee.ethz.ch/yawli/shard-08.tar.gz
https://data.vision.ee.ethz.ch/yawli/shard-09.tar.gz
https://data.vision.ee.ethz.ch/yawli/shard-10.tar.gz
https://data.vision.ee.ethz.ch/yawli/shard-11.tar.gz
https://data.vision.ee.ethz.ch/yawli/shard-12.tar.gz
https://data.vision.ee.ethz.ch/yawli/shard-13.tar.gz
https://data.vision.ee.ethz.ch/yawli/shard-14.tar.gz
https://data.vision.ee.ethz.ch/yawli/shard-15.tar.gz
https://data.vision.ee.ethz.ch/yawli/shard-16.tar.gz

# Download script:

```bash
#!/bin/bash

while IFS= read -r line
do
    if [[ $line == http* ]]
    then
        wget "$line"
    fi
done < README.md
```

# Extract script:

```bash
#!/bin/bash=

# Directory containing the .tar.gz files
TARGET_DIR="."

# Change to the target directory
cd "$TARGET_DIR" || exit

# Loop through all .tar.gz files and uncompress them
for FILE in *.tar.gz; do
    if [ -f "$FILE" ]; then
        echo "Uncompressing $FILE..."
        tar -xzvf "$FILE"
    else
        echo "No .tar.gz files found in $TARGET_DIR"
        exit 1
    fi
done
```

# Crop images script:

use the crop.py script to crop the images
