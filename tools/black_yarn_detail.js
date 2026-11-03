const $ = id => document.getElementById(id);
const state = {items: [], index: 0, line: [], dirty: false, zoom: 1};

function cur() { return state.items[state.index]; }
function msg(value, error = false) {
  $('message').textContent = value;
  $('message').className = 'message' + (error ? ' error' : '');
}
function dirty() { state.dirty = true; msg('尚未保存'); }
function list() {
  $('count').textContent = `${state.items.filter(x => x.detail).length} / ${state.items.length}`;
  $('list').replaceChildren();
  state.items.forEach((item, index) => {
    const button = document.createElement('button');
    button.className = 'item' + (index === state.index ? ' active' : '');
    button.innerHTML = '<span class="dot"></span><b></b><small></small>';
    button.querySelector('b').textContent = item.frame_id;
    button.querySelector('small').textContent = `${item.cam} · ${item.status === 'black_yarn' ? '可见黑纱' : '遮挡待定'}`;
    if (item.detail) button.querySelector('.dot').classList.add('done');
    button.onclick = () => go(index);
    $('list').append(button);
  });
}
function info() { $('shapeInfo').textContent = `黑纱中心线 ${state.line.length} 点`; }
function show() {
  const item = cur(), detail = item.detail || {};
  state.line = (detail.yarn_centerline || []).map(point => [...point]);
  state.zoom = 1;
  state.dirty = false;
  $('photo').removeAttribute('src');
  $('photo').style.width = '100%';
  $('photo').src = `/image/${encodeURIComponent(item.frame_id)}`;
  $('name').textContent = item.frame_id;
  $('meta').textContent = `${state.index + 1} / ${state.items.length} · ${item.cam} · ${item.status}`;
  $('notes').value = detail.notes || '';
  $('instruction').textContent = item.status === 'black_yarn'
    ? '只画可见的黑纱，不画手。至少点击 2 点，然后保存。'
    : '此图被遮挡，不能确定黑纱。只写原因，不画猜测线。';
  msg(item.detail ? '已保存缺陷复核草稿' : '尚未复核');
  info();
  list();
}
function resize() {
  const image = $('photo'), canvas = $('overlay');
  if (!image.naturalWidth) return;
  const ratio = devicePixelRatio || 1;
  canvas.style.width = image.clientWidth + 'px';
  canvas.style.height = image.clientHeight + 'px';
  canvas.width = Math.round(image.clientWidth * ratio);
  canvas.height = Math.round(image.clientHeight * ratio);
  draw();
}
function draw() {
  const image = $('photo'), canvas = $('overlay');
  if (!image.naturalWidth) return;
  const ctx = canvas.getContext('2d'), ratio = devicePixelRatio || 1;
  const sx = image.clientWidth / image.naturalWidth, sy = image.clientHeight / image.naturalHeight;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, image.clientWidth, image.clientHeight);
  for (const box of cur().a_boxes) {
    ctx.strokeStyle = '#19b979';
    ctx.lineWidth = 3;
    ctx.strokeRect(box[0] * sx, box[1] * sy, (box[2] - box[0]) * sx, (box[3] - box[1]) * sy);
  }
  if (!state.line.length) return;
  ctx.beginPath();
  ctx.moveTo(state.line[0][0] * sx, state.line[0][1] * sy);
  for (const point of state.line.slice(1)) ctx.lineTo(point[0] * sx, point[1] * sy);
  ctx.strokeStyle = '#39c8de';
  ctx.lineWidth = 3;
  ctx.stroke();
  for (const point of state.line) {
    ctx.beginPath();
    ctx.arc(point[0] * sx, point[1] * sy, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#39c8de';
    ctx.fill();
  }
}
$('overlay').onclick = event => {
  if (!$('photo').naturalWidth || cur().status !== 'black_yarn') return;
  const rect = $('overlay').getBoundingClientRect(), image = $('photo');
  state.line.push([
    Math.round(Math.max(0, Math.min(image.naturalWidth, (event.clientX - rect.left) * image.naturalWidth / rect.width))),
    Math.round(Math.max(0, Math.min(image.naturalHeight, (event.clientY - rect.top) * image.naturalHeight / rect.height)))
  ]);
  dirty(); info(); draw();
};
async function save() {
  const item = cur();
  if (state.line.length === 1) { msg('黑纱中心线至少需要 2 点', true); return false; }
  try {
    const response = await fetch(`/api/items/${encodeURIComponent(item.frame_id)}`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({yarn_centerline: state.line, notes: $('notes').value})
    });
    const data = await response.json();
    if (!response.ok) throw Error(data.error || '保存失败');
    item.detail = {yarn_centerline: state.line.map(point => [...point]), notes: $('notes').value,
                   reviewed_at_utc: data.reviewed_at_utc};
    state.dirty = false;
    msg('已保存本机缺陷复核草稿');
    list();
    return true;
  } catch (error) { msg(error.message, true); return false; }
}
async function go(index) {
  if (index < 0 || index >= state.items.length || index === state.index) return;
  if (state.dirty && !(await save())) return;
  state.index = index; show();
}
function zoom(value) {
  state.zoom = Math.min(4, Math.max(1, value));
  $('photo').style.width = `${state.zoom * 100}%`;
  requestAnimationFrame(resize);
}
$('photo').onload = resize;
new ResizeObserver(resize).observe($('photo'));
$('undo').onclick = () => { if (state.line.length) { state.line.pop(); dirty(); info(); draw(); } };
$('clearYarn').onclick = () => { state.line = []; dirty(); info(); draw(); };
$('notes').oninput = dirty;
$('save').onclick = save;
$('prev').onclick = () => go(state.index - 1);
$('next').onclick = () => go(state.index + 1);
$('in').onclick = () => zoom(state.zoom + .5);
$('out').onclick = () => zoom(state.zoom - .5);
$('zoom').onclick = () => zoom(1);
document.addEventListener('keydown', event => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
    event.preventDefault(); save();
  } else if (!['TEXTAREA', 'INPUT'].includes(document.activeElement.tagName)) {
    if (event.key === 'ArrowRight') go(state.index + 1);
    if (event.key === 'ArrowLeft') go(state.index - 1);
  }
});
fetch('/api/items').then(response => response.json()).then(items => {
  if (!Array.isArray(items)) throw Error(items.error || '载入失败');
  state.items = items; show();
}).catch(error => { msg(error.message, true); $('name').textContent = '读取失败'; });
