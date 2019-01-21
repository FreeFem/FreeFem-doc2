// Remove first ul - li (name of the current part)
function adjustToc() {
    const tocLayout = document.getElementById('toc')
    const tocSubLayout = tocLayout.children[0]
    const toc = tocSubLayout.children[0]
    const topItem = toc.children[0] //Already exists

    if (topItem.children && topItem.children.length > 1) {
        const subItem = topItem.children[1];

        toc.removeChild(topItem)
        toc.appendChild(subItem)
    } else {
        toc.innerHTML = ''
    }
}

adjustToc()
