var path = require("path");
var fs = require("fs");
var lunr = require("lunr");
var cheerio = require("cheerio");

const HTML_FOLDER = "../build/html";
const SEARCH_FIELDS = ["title", "description", "keywords", "body"];
const EXCLUDE_FILES = ["search.html", "index.html", "genindex.html"];
const MAX_PREVIEW_CHARS = 275;
const OUTPUT_INDEX = "lunr_index.js";

function isHtml(filename) {
    lower = filename.toLowerCase();
    return (lower.endsWith(".htm") || lower.endsWith(".html"));
}

function findHtml(folder) {
    if (!fs.existsSync(folder)) {
        console.log("Could not find folder: ", folder);
        return;
    }

    var files = fs.readdirSync(folder);
    var htmls = [];
    for (var i = 0; i < files.length; i++) {
        var filename = path.join(folder, files[i]);
        var stat = fs.lstatSync(filename);
        if (stat.isDirectory()) {
            var recursed = findHtml(filename);
            for (var j = 0; j < recursed.length; j++) {
                recursed[j] = path.join(files[i], recursed[j]).replace(/\\/g, "/");
            }
            htmls.push.apply(htmls, recursed);
        }
        else if (isHtml(filename) && !EXCLUDE_FILES.includes(files[i])) {
            htmls.push(files[i]);
        };
    };
    return htmls;
};

function readHtml(root, file, fileId) {
    var filename = path.join(root, file);
    var txt = fs.readFileSync(filename).toString();

    var $ = cheerio.load(txt, {normalizeWhiteSpace: true, decodeEntities: true});
    var title = $('title').text()

    // var data = []
    // $('.section').each(function(i, elem) {
    //     if ($(this).children('.section').length === 0) {    //this is the deeper section
    //         let subTitle = ''
    //         if ($(this).children('h1').length) {
    //             subTitle = $(this).children('h1').text()
    //         } else if ($(this).children('h2').length) {
    //             subTitle = $(this).children('h2').text()
    //         } else if ($(this).children('h3').length) {
    //             subTitle = $(this).children('h3').text()
    //         } else if ($(this).children('h4').length) {
    //             subTitle = $(this).children('h4').text()
    //         } else if ($(this).children('h5').length) {
    //             subTitle = $(this).children('h5').text()
    //         } else {
    //             console.error('INTERNAL ERROR')
    //             console.error('No title')
    //         }
    //
    //         subTitle = subTitle.replace('¶', '')
    //
    //         var anchor = subTitle.toLowerCase()
    //         while(anchor.indexOf(' ') > -1)
    //             anchor = anchor.replace(' ', '-')
    //         while(anchor.indexOf('/') > -1)
    //             anchor = anchor.replace('/', '-')
    //         while(anchor.indexOf('+') > -1)
    //             anchor = anchor.replace('+', '')
    //         while(anchor.indexOf('\'') > -1)
    //             anchor = anchor.replace(' ', '')
    //
    //         var link = file + '#' + anchor
    //         var body = $(this).text()
    //         if (title === undefined) title = file;
    //
    //         data.push({
    //             "id": fileId,
    //             "link": link,
    //             "t": title + " " + subTitle,
    //             "d": "",
    //             "k": "",
    //             "b": body
    //         })
    //     }
    // })

    if (typeof title == 'undefined') title = file;
    var description = $("meta[name=description]").attr("content");
    if (typeof description == 'undefined') description = "";
    var keywords = $("meta[name=keywords]").attr("content");
    if (typeof keywords == 'undefined') keywords = "";
    var body = $("body").text()
    if (typeof body == 'undefined') body = "";
    var data = [{
        "id": fileId,
        "link": file,
        "t": title,
        "d": description,
        "k": keywords,
        "b": body
    }]

    return data;
}

function buildIndex(docs) {
    var idx = lunr(function () {
        this.ref('id');
        for (var i = 0; i < SEARCH_FIELDS.length; i++) {
            this.field(SEARCH_FIELDS[i].slice(0, 1));
        }
        docs.forEach(function (doc) {
                this.add(doc);
            }, this);
        });
    return idx;
}

function buildPreviews(docs) {
    var result = {};
    for (var i = 0; i < docs.length; i++) {
        var doc = docs[i];
        var preview = doc["d"];
        if (preview == "") preview = doc["b"];
        // if (preview.length > MAX_PREVIEW_CHARS)
        //     preview = preview.slice(0, MAX_PREVIEW_CHARS) + " ...";
        result[doc["id"]] = {
            "t": doc["t"],
            "p": preview,
            "l": doc["link"]
        }
    }
    return result;
}

function main() {
    files = findHtml(HTML_FOLDER);
    var docs = [];
    console.log("Building index for these files:");
    var j = 0;
    for (var i = 0; i < files.length; i++) {
        console.log("    " + files[i]);
        docs = docs.concat(readHtml(HTML_FOLDER, files[i], j++));
    }
    var idx = buildIndex(docs);
    var previews = buildPreviews(docs);
    var js = "const LUNR_DATA = " + JSON.stringify(idx) + ";\n" +
             "const PREVIEW_LOOKUP = " + JSON.stringify(previews) + ";";
    fs.writeFile(OUTPUT_INDEX, js, function(err) {
        if(err) {
            return console.log(err);
        }
        console.log("Index saved as " + OUTPUT_INDEX);
    });
}

main();
