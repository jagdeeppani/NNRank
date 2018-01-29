ssh -o IdentitiesOnly=yes -o "ServerAliveInterval 60" -i ~/community-cluster.pem pjagdeep@ec2-34-207-229-83.compute-1.amazonaws.com


HADOOP_INSTALL=/usr/
dumbo start RunNMF.py -libjar feathers.jar -hadoop $HADOOP_INSTALL -projsize 12 -reduce_schedule 2,1 -mat SmallNoisySep_10k_10_4.txt -output small.out -hadooplib /usr/lib/hadoop/


dumbo cat small.out/GP -hadoop $HADOOP_INSTALL -hadooplib /usr/lib/hadoop/ > small.out-proj.txt
dumbo cat small.out/QR -hadoop $HADOOP_INSTALL -hadooplib /usr/lib/hadoop/ > small.out-qrr.txt
dumbo cat small.out/colnorms -hadoop $HADOOP_INSTALL -hadooplib /usr/lib/hadoop/ > small.out-colnorms.txt

python NMFProcessAlgorithms.py small.out-qrr.txt small.out-colnorms.txt 'SPA' 4
python NMFProcessAlgorithms.py small.out-qrr.txt small.out-colnorms.txt 'xray' 4
python NMFProcessAlgorithms.py small.out-proj.txt small.out-colnorms.txt 'GP' 4



hadoop fs -put data/FC_40k.txt FC_40k.txt

HADOOP_INSTALL=/usr/
dumbo start FC_kron.py -mat FC_40k.txt -output FC_kron.bseq -hadoop $HADOOP_INSTALL -hadooplib /usr/lib/hadoop/


HADOOP_INSTALL=/usr/
dumbo start RunNMF.py -libjar feathers.jar -hadoop $HADOOP_INSTALL -reduce_schedule 40,1 -mat FC_kron.bseq -output FC_data.out -hadooplib /usr/lib/hadoop/

dumbo cat FC_data.out/GP -hadoop $HADOOP_INSTALL -hadooplib /usr/lib/hadoop/ > FC_data.out-proj.txt
dumbo cat FC_data.out/QR -hadoop $HADOOP_INSTALL -hadooplib /usr/lib/hadoop/ > FC_data.out-qrr.txt
dumbo cat FC_data.out/colnorms -hadoop $HADOOP_INSTALL -hadooplib /usr/lib/hadoop/ > FC_data.out-colnorms.txt

python NMFProcessAlgorithms.py FC_data.out-qrr.txt FC_data.out-colnorms.txt 'SPA' 5
python NMFProcessAlgorithms.py FC_data.out-qrr.txt FC_data.out-colnorms.txt 'xray' 5
python NMFProcessAlgorithms.py FC_data.out-proj.txt FC_data.out-colnorms.txt 'GP' 5


#---------------------------
#Apply mrnmf on age data
#---------------------------










hduser@hadoop-PC:~/hadoop$./bin/hadoop jar contrib/streaming/hadoop-streaming-1.1.2.jar** -file /home/hduser/mapper.py    -mapper /home/hduser/mapper.py -file /home/hduser/reducer.py   -reducer /home/hduser/reducer.py -input /user/hduser/gutenberg/* -output /user/hduser/gutenberg-output


echo $HADOOP_HOME/contrib/streaming/hadoop-streaming-*.jar


HADOOP_HOME="/usr/lib/hadoop"
STREAMING_JAR="`echo $HADOOP_HOME/contrib/streaming/hadoop-*-streaming.jar`"


/usr/lib/hadoop/hadoop-streaming-2.7.3-amzn-1.jar
/usr/lib/hadoop/hadoop-streaming.jar

-hadooplib /usr/lib/hadoop/

grep -rnw './' -e 'ERROR: Streaming jar not found'
grep -rnw './' -e 'ERROR: Streaming jar not found'

grep -rnw './' -e 'dumbo'


/usr/lib/python2.7/dist-packages/

dumbo-master/build/lib/dumbo/backends/streaming.py
dumbo-master/dumbo/backends/streaming.py

streamingjar = '/usr/lib/hadoop/hadoop-streaming.jar'

from dumbo.util import (configopts, envdef, execute, findhadoop, findjar,
        dumpcode, dumptext, Options)
        
streamingjar = findjar(self.hadoop, 'streaming',
                       opts['hadooplib'] if 'hadooplib' in opts else None)
if not streamingjar:
    print >> sys.stderr, 'ERROR: Streaming jar not found'
    return 1

streamingjar = findjar(hadoop, 'streaming', addedopts['hadooplib'])
if not streamingjar:
    print >> sys.stderr, 'ERROR: Streaming jar not found'
    return 1
    
    
    
    
    
    
    
#build.sh in feathers    
    
#!/bin/sh

HADOOP_HOME="/usr/lib/hadoop"

HADOOP_JAR="`echo $HADOOP_HOME/hadoop-common*.jar`"
STREAMING_JAR="`echo $HADOOP_HOME/hadoop-streaming*.jar`"

#if STREAMING_JAR==""; then
#    STREAMING_JAR="`echo $HADOOP_HOME/contrib/streaming/hadoop-streaming-*.jar`"
#fi

rm -rf classes 2> /dev/null
rm feathers.jar 2> /dev/null
if test "$1" = clean; then exit; fi

mkdir classes
javac -classpath "$HADOOP_JAR:$STREAMING_JAR" -d classes src/*/*.java
jar -cvf feathers.jar -C classes/ .

javac -cp $(hadoop classpath) -d classes src/*/*.java

#replace get() by getBytes() 
#replace getSize() by getLength()









