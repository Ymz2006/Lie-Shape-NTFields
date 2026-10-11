"""Serve the experiment tree so it can be viewed from a laptop without downloading it.

    python3 research_tree/serve.py               # http://localhost:8765 on the server
    python3 research_tree/serve.py --port 9000

tree.yaml is re-validated and re-rendered on every page load, so refreshing the
browser always shows the current tree. The server binds to 127.0.0.1 by default;
reach it from your laptop with an SSH tunnel (the command is printed on start):

    ssh -N -L 8765:localhost:8765 <user>@<server>     # then open http://localhost:8765

`--host 0.0.0.0` exposes it on the network instead (no tunnel, no auth).
"""
import argparse
import getpass
import http.server
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render  # noqa: E402


class Handler(http.server.BaseHTTPRequestHandler):
    tree = os.path.join(render.HERE, 'tree.yaml')

    def do_GET(self):
        if self.path.split('?')[0] not in ('/', '/index.html', '/tree.html'):
            self.send_error(404)
            return
        try:
            meta, nodes = render.load(self.tree)
            out = os.path.join(render.HERE, 'tree.html')
            render.write_html(meta, nodes, out)
            with open(out, 'rb') as f:
                body, code, ctype = f.read(), 200, 'text/html; charset=utf-8'
        except SystemExit as e:  # render.load exits with the validation errors
            body, code, ctype = str(e).encode(), 500, 'text/plain; charset=utf-8'
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--host', default='127.0.0.1')
    args = ap.parse_args()

    srv = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    print('Serving research_tree/tree.html on http://{}:{}'.format(args.host, args.port))
    if args.host in ('127.0.0.1', 'localhost'):
        print('From your laptop:\n  ssh -N -L {0}:localhost:{0} {1}@{2}\n  then open http://localhost:{0}'.format(
            args.port, getpass.getuser(), socket.getfqdn()))
        print('(VS Code Remote-SSH users: just open the forwarded port it offers.)')
    print('Ctrl+C to stop.', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
