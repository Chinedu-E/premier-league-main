
async function getData() {
  const response = await fetch("predictions.csv");
  var data = await response.text();
  data = data.split("\n");
  const headers = data[0];
  console.log(headers)
  const rows = data.slice(1);
  console.log(rows)
  return [headers, rows];
}

async function tableCreate() {
  const response = await fetch("predictions.csv");
  var data = await response.text();
  data = data.split("\n");
  const headers = data[0].split(",");
  const rows = data.slice(1);
  var body = document.getElementsByClassName("myTable")[0];
  var tbl = document.createElement("table");
  tbl = styleTable(tbl)
  var thead = document.createElement('thead');
  var tr_head = document.createElement('tr');
  var tbdy = document.createElement('tbody');

  for (var i = 0; i < headers.length; i++) {
    var th = document.createElement('th')
    th.appendChild(document.createTextNode(headers[i]))
    tr_head.appendChild(th)
    }
  thead.appendChild(tr_head)
  tbl.appendChild(thead)

  for (var i = 0; i < rows.length; i++) {
    var tr = document.createElement('tr');
    var cols = rows[i].split(",");
    var probs = cols.slice(-3);
    max_idx = findMax(probs)
    for (var j = 0; j < cols.length; j++) {
        var td = document.createElement('td');
        if (j == max_idx+4) {
          td.setAttribute("class", "table-success")
        }
        td.appendChild(document.createTextNode(`${cols[j]}`))
        tr.appendChild(td)
    }
    tbdy.appendChild(tr);
  }
  tbl.appendChild(tbdy);
  body.appendChild(tbl);
}

function styleTable(table){
  table.style.width = '70%';
  table.style.marginLeft = "auto";
  table.style.marginRight = "auto";
  table.setAttribute('border', '2');
  table.setAttribute('class', "table table-bordered table-hover table-dark");
  return table;
}

function findMax(arr) {
  var max_idx = 0
  var max_prob = 0.0
  for (var i=0; i<arr.length; i++){
    arr[i] = parseFloat(arr[i])
    if (arr[i] > max_prob) {
      max_idx = i
      max_prob = arr[i]
    }
  }
  return max_idx
}

tableCreate();