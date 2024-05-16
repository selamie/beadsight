import socket
import argparse
import time

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        # doesn't even have to be reachable
        s.connect(('8.8.8.8', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

class RemoteInput:
    def __init__(self, host: bool, host_address: str = None, port: int = 12345):
        self.host = host
        
        if self.host:
            # get the host's ip address
            host_address = get_ip()
            host_address = '127.0.0.1'
            print(f"Host address: {host_address}")
            # Create a socket object
            self.socket = socket.socket()
            # Bind to the port, check if the default port is already in use
            old_port = port
            while True:
                try:
                    self.socket.bind((host_address, port))
                    break
                except OSError:
                    try_port += 1
            
            if old_port != port:
                print(f"Port {old_port} was already in use. Using port {port} instead.")

            # Now wait for client connection.
            self.socket.listen()
            print(f"Listening on port {port}...")
            # Establish connection with client.
            self.connection, self.client_address = self.socket.accept()
            print(f"Connection from {self.client_address}")
        
        else: # connect to the host

            if host_address is None:
                host_address = input("Enter the host's ip address: ")
            host_address = '127.0.0.1'

            new_port = input(f"Enter the host's port, or press enter to use the default ({port}): ")
            if new_port != "":
                port = int(new_port)

            self.socket = socket.socket()
            self.socket.connect((host_address, port))
            print(f"Connected to {host_address} on port {port}.")
            

            self.get_user_input()               


    def input(self, message: str) -> str:
        # encdoe the message as bytes.
        # The first 4 bytes are the length of the message (including the type byte)
        # The next byte is b'i' to indicate that this is an input message
        # The rest of the bytes are the message
        print('begin input function')
        message_data = (len(message) + 1).to_bytes(4, 'big') + b'i' + message.encode()
        self.socket.send(message_data)
        print('message sent')

        # wait for the input to be received
        recieved_buffer = self.socket.recv(1024)
        if b'\x04' in recieved_buffer: # end of stream
            print("Connection closed by host (EoT character).")
            self.connection.close()
            self.socket.close()
            return None
        
        message_size = int.from_bytes(recieved_buffer[:4], 'big')
        recieved_buffer = recieved_buffer[4:] # clear the header from the buffer
        while len(recieved_buffer) < message_size:
            recieved_buffer += self.socket.recv(1024)
        
        assert len(recieved_buffer) == message_size, f"Expected {message_size} bytes, but got {len(recieved_buffer)} bytes."

        # decode the message
        return recieved_buffer.decode() 
        
    
    def print(self, message: str):
        # encdoe the message as bytes.
        # The first 4 bytes are the length of the message (including the type byte)
        # The next byte is p to indicate that this is a print message
        # The rest of the bytes are the message
        message_data = (len(message) + 1).to_bytes(4, 'big') + b'p' + message.encode()
        self.socket.send(message_data)


    def get_user_input(self):
        recieved_buffer = bytes()
        while True:
            # pull data from the socket
            print(self.socket.getblocking(), self.socket.gettimeout())
            new_data = self.socket.recv(1024)
            if new_data == b'':
                time.sleep(0.1)
                continue
            print('new data: ', new_data)
            if b'\x04' in new_data: # end of stream
                print("Connection closed by host (EoT character).")
                self.socket.close()
                break

            recieved_buffer += new_data

            # decode the message
            # The first 4 bytes are the length of the message
            message_size = int.from_bytes(recieved_buffer[:4], 'big')
            recieved_buffer = recieved_buffer[4:] # clear the header from the buffer

            while len(recieved_buffer) < message_size:
                recieved_buffer += self.socket.recv(1024)
            
            # decode the message
            message_type = recieved_buffer[0:1].decode()
            message = recieved_buffer[1:message_size].decode()
            recieved_buffer = recieved_buffer[message_size:] # clear the message from the buffer
            

            # print or input
            if message_type == 'i': # input
                input_message = self.input(message).encode()
                # send the input back to the host (begining with the size of the message)
                self.socket.send(len(input_message).to_bytes(4, 'big') + input_message)

            if message_type == 'p': # print
                print(message)
            
            else:
                raise ValueError(f"Unknown message type: {message_type}")
        
    
    def close(self):
        if self.host:
            self.socket.send(b'\x04')
            self.connection.close()
            self.socket.close()
        else:
            self.socket.send(b'\x04')
            self.socket.close()

    # close the process when the object is deleted
    def __del__(self):
        self.close()

    # allow the object to be used as a context manager (with statement)
    def __enter__(self):
        return self

    # close the process when the context manager is exited (exit with statement)
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == '__main__': 
    # By default, the user will try to connect to a host
    # IP address and port can be specified with the --host_address and --port arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default=None, help='The host\'s IP address.', required=False)
    parser.add_argument('--port', type=int, default=12345, help='The port to use.', required=False)

    args = parser.parse_args()
    RemoteInput(host=False, host_address=args.host, port=args.port)