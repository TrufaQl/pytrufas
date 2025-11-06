import mysql.connector as sql
# ========== ========== ========== ========== ========== #
class DataBase():
    def __init__(self, host, user, password, name):
        try:
            conn = sql.connect(
                host=host,
                user=user,
                password=password
            )

            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {name}")
            cursor.close()
            conn.close()
        except Exception as e:
            print("----- Create DataBase -----")
            print(f"Error: {e}")
            return
        self.host = host
        self.user = user
        self.password = password
        self.name = name
    
    def connectSQL(self):
        try:
            conn = sql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.name
            )
            self.cursor = conn.cursor()
            self.cursor.execute("SELECT DATABASE()")
            self.cursor.fetchone()
        except Exception as e:
            print("----- Connect SQL -----")
            print(f"Error: {e}")

    def execute(self, querys):
        resultados = []
        for query in querys:
            self.cursor.execute(query)
            resultado = self.cursor.fetchall()
            resultados.append(resultado)
        self.conn.commit()
        return resultados
# ========== ========== ========== ========== ========== #
