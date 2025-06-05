import oracledb

# Initialize client (only for thick mode)
oracledb.init_oracle_client(lib_dir=r"D:\Oracle21c\instantclient-basic-windows.x64-23.8.0.25.04\instantclient_23_8")

# Connect
connection = oracledb.connect(
    user="system",
    password="user",  # make sure this is the correct password
    dsn="192.168.50.21//XEPDB1"
)

cursor = connection.cursor()
cursor.execute("SELECT table_name FROM user_tables")  # Check available tables
for row in cursor:
    print(row)

cursor.close()
connection.close()
