import socket

class mysocket:
    '''class socket
       class to handle socket server connection
    '''

    def __init__(self):
        self.s = socket.socket()
        #self.host = socket.gethostname()
        self.host = '192.168.178.27'
        self.port = 11000

    def connect(self):
        self.s.connect((self.host, self.port))

    def send(self, msg):
        self.s.send(msg + '\r')

    def receive(self):
        reply = self.s.recv(1024)
        return reply
