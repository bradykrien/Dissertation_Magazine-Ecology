## Overview

This project involves the utilization of the HathiTrust's [Extracted Features 2.0 dataset] (https://wiki.htrc.illinois.edu/pages/viewpage.action?pageId=79069329), a non-consumptive dataset that contains most of the textual data used in this project. 

The data for this project are downloaded from the HTRC as zipped json.bz2 files through rsync. These files can then be acccessed and manipulated as dataframes in jupyter notebooks or other formats. 

### Getting the IDs
In order to access the Extracted Features dataset, the first thing that I did was to build a [collection](https://www.hathitrust.org/htrc_collections_tools) on the HTRC site. Once I had built a [collection](https://babel.hathitrust.org/cgi/mb?a=listis&c=68556871), I downloaded the the metadata Linked Data as a JSON file. I then removed the overall collection metadata and changed the first organization level to "texts" (see the SampleMetaData.json for exact formatting) to ease the conversion process using the short JSONconverter.py script, which writes a CSV file with the necessary HathiTrust identifiers (HTIDs). 

### Create the Pathlist
Once I had the HTIDs, I used used the [conversion identification function in the HTRC feature-reader](https://wiki.htrc.illinois.edu/display/COM/Finding+Extracted+Features+data+for+a+known+volume+ID) to generate a pathlist. The specific process for this can be found in the "Rsync Pathlist Generator" jupyter notebook. This generates a CSV pathlist to be used for the rsync process. 

### Downloading the Files
After moving the pathlist.csv file to the directory to which you want the EF files downloaded, the final step was to run the command: 

rsync -av --no-relative --files-from=pathlist.txt data.analytics.hathitrust.org::features-2020.03/ .

This downloads the .json.bz2 files for each of the volumes in the collection. These can then be manipulated and analyzed using the htrc-feature-reader library. Peter Organisciak and Boris Capitanu have a great introductory tutorial to the library and how to use it in their ["Text Mining in Python through the HTRC Feature Reader"](https://programminghistorian.org/en/lessons/text-mining-with-extracted-features) lesson on the [Programming Historian](https://programminghistorian.org/)
