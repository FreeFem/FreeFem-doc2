window.onclick = function(e) {
    if (!e.target.matches('#searchResults') && !e.target.matches('#resultCount')) {
        const searchResults = document.getElementById('searchResults')
        const resultCount = document.getElementById('resultCount')

        searchResults.innerHTML = ''
        resultCount.innerHTML = ''
    }

    if (!e.target.matches('.dropdown') && !(e.target.parentNode && e.target.parentNode.matches('.dropdown'))) {
        const dropdownContent = document.getElementsByClassName('dropdown-content')
        for (let i = 0; i < dropdownContent.length; i++) {
            dropdownContent[i].style.display = 'none'
        }
    }
}
