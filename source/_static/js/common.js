window.onclick = function(e) {
   if (!e.target.matches('#searchResults')) {
      const searchResults = document.getElementById('searchResults')
      searchResults.style.display = 'none'
   }
}
