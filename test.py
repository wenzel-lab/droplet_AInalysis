class MiClase:
    def __init__(self, otro:str, valor='valor_por_defecto'):
        self.otro = otro
        self.valor = valor

# Crear una instancia sin proporcionar un argumento
objeto1 = MiClase("a")
print(objeto1.valor)  # Salida: valor_por_defecto

# Crear una instancia proporcionando un argumento
objeto2 = MiClase("a", 'otro_valor')
print(objeto2.valor)  # Salida: otro_valor
