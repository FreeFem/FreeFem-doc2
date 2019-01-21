function goto(url) {
    window.location = url
}

function nav(root) {
    // Nav
    document.write('<div class="dropdown">')
    for (let i = 0; i < navData.length; i++) {
        if (navData[i].children) {
            const title = navData[i].title
            document.write('<div class="dropdown-header">')
            document.write('<button class="dropdown" data-target="dropdown'+i+'" onclick="dropdown(this);">'+title+'<span class="separator"></span><i class="fas fa-angle-down"></i></button>')
            const path = navData[i].path
            document.write('<div id="dropdown'+i+'" class="dropdown-content">')
            const keys = Object.keys(navData[i].children)
            const values = Object.values(navData[i].children)
            for (let j = 0; j < keys.length; j++) {
                const url = root + path + '/' + keys[j] + '.html'
                document.write('<button type="button" onclick="goto(\''+url+'\');">'+values[j]+'</button>')
            }
            document.write('</div>')
            document.write('</div>')
        } else {
            const title = navData[i].title
            const path = navData[i].path
            const file = navData[i].file
            const url = root + path + '/' + file + '.html'
            document.write('<div class="dropdown-header"><button type="button" onclick="goto(\''+url+'\')">'+title+'</button></div>')
        }
    }
    document.write('</div>')
}

function dropdown(elmt) {
    const dropdownContent = document.getElementsByClassName('dropdown-content')
    for (let i = 0; i < dropdownContent.length; i++) {
        dropdownContent[i].style.display = 'none'
    }

    const subElmtId = elmt.getAttribute('data-target')
    const subElmt = document.getElementById(subElmtId)
    subElmt.style.display = 'flex'
}
