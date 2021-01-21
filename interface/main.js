const { app, BrowserWindow, Menu, ipcMain, dialog } = require('electron')
const fs = require('fs')
const path = require('path')
const { PythonShell } = require('python-shell')

/*let options = {
    mode: 'text',
    pythonPath: 'python',
    pythonOptions: ['-u'],
    scriptPath: '',
    args: []
}
PythonShell.run('engine.py', options, function(err) {
    if(err) console.log(err)
})*/

function createWindow() {
    const win = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            nodeIntegration: true,
            enableRemoteModule: true
        }
    })
    win.loadFile('index.html')
    const menu = Menu.buildFromTemplate([
        {
            label: 'File',
            submenu: [
                {
                    label: 'Open',
                    click: () => {
                        const result = dialog.showOpenDialogSync(win, {
                            properties: ['openFile']
                        })
                        if(result) {
                            win.webContents.send('loadConfig', result[0])
                        }
                    }
                }
            ]
        },
        {
            label: 'View',
            submenu: [
                { role: 'reload' },
                { role: 'forceReload' },
                { role: 'toggleDevTools' }
            ]
        }
    ])
    Menu.setApplicationMenu(menu)
}

app.whenReady().then(createWindow)

app.on('window-all-closed', () => {
    if(process.platform !== 'darwin') {
        app.quit()
    }
})

app.on('activate', () => {
    if(BrowserWindow.getAllWindows().length === 0) {
        createWindow()
    }
})
