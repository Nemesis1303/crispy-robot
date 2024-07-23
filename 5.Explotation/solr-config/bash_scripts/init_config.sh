FILE=""
DIR="$1"
# init
# look for empty dira
if [ -d "$DIR" ]
then
	if [ "$(ls -A $DIR)" ]; then
        echo "$DIR is NOT Empty"
        echo "No Solr Config Initialization Required"
	else
    echo "$DIR is Empty"
    echo "---> Inititalizing Solr Config..."
    docker exec into-solr bin/solr zk upconfig -zkhost zoo:2181 -n into_config -d /opt/solr-9.1.1/server/solr/configsets/into_config
	fi
else
	echo "Directory $DIR not found."
    echo "---> Inititalizing Solr Config..."
    docker exec into-solr bin/solr zk upconfig -zkhost zoo:2181 -n into_config -d /opt/solr-9.1.1/server/solr/configsets/into_config
fi





